#!/usr/bin/env python
"""Run a legacy WebSocket end-to-end flow with seeded QBR data."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
from penguiflow.planner import PlannerFinish
from sqlalchemy import JSON, Column, Float, Integer, MetaData, String, Table, Text, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_agent_qbr.api.app import create_app
from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.models import Query
from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
from ai_agent_qbr.tools.analyze import analyze_results
from ai_agent_qbr.tools.search import search_documents
from ai_agent_qbr.transport.websocket.schemas import (
    OutputError,
    OutputFinal,
    OutputPing,
    OutputPong,
    OutputReady,
    OutputThinking,
    OutputUserMessage,
)
from qbr_agent.application.ports import Reranker
from qbr_agent.infrastructure.embeddings import HashEmbeddingsProvider
from qbr_agent.infrastructure.factory import InfrastructureBundle
from qbr_agent.infrastructure.sqlalchemy_repository import SqlAlchemyKnowledgeRepository
from qbr_agent.infrastructure.vector_index import SqliteEmbeddingVectorIndex


QUERY = "Hi, can you show me difference in added value regarding h1 and h2"


class ToolRunnerPlanner:
    def __init__(self) -> None:
        self.last_llm_context = None
        self.last_search_results = None

    async def run(self, *, query, llm_context, tool_context):
        self.last_llm_context = llm_context
        ctx = type("ToolContext", (), {"tool_context": tool_context})
        search_results = await search_documents(Query(question=query), ctx)
        self.last_search_results = search_results
        analysis = await analyze_results(search_results, ctx)
        return PlannerFinish(
            reason="answer_complete",
            payload={"raw_answer": analysis.text},
            metadata={},
        )


@dataclass
class DebugReranker(Reranker):
    calls: int = 0

    async def score(self, *, query: str, chunks) -> list[float]:
        self.calls += 1
        needle = query.lower()
        scores = []
        for chunk in chunks:
            text = chunk.content.lower()
            score = 0.0
            if "h1" in text:
                score += 2.0
            if "h2" in text:
                score += 2.0
            if "added value" in text:
                score += 1.0
            if "difference" in needle or "difference" in text:
                score += 0.5
            scores.append(score)
        return scores


def _setup_db(db_url: str) -> None:
    async def _run():
        engine = create_async_engine(db_url)
        metadata = MetaData()
        Table(
            "documents",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("filename", String(255)),
            Column("client_name", String(255)),
            Column("period", String(100)),
            Column("status", String(50)),
            Column("executive_summary", Text),
        )
        Table(
            "chunks",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("document_id", Integer),
            Column("content", Text),
            Column("start_slide", Integer),
            Column("end_slide", Integer),
            Column("summary", Text),
            Column("topics", JSON),
            Column("importance_score", Float),
            Column("embedding", JSON),
            Column("embedding_model", String(100)),
        )
        Table(
            "slides",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("document_id", Integer),
            Column("slide_number", Integer),
            Column("title", Text),
            Column("key_message", Text),
            Column("slide_type", String(100)),
            Column("insights", Text),
            Column("action_items", Text),
        )

        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            try:
                await conn.execute(
                    text(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
                        "USING fts5(content, content='chunks', content_rowid='id')"
                    )
                )
            except Exception:
                pass

            provider = HashEmbeddingsProvider()
            chunks = [
                (1, 1, "H1 added value increased by 12% vs baseline.", 2),
                (2, 1, "H1 vs H2 difference highlights stronger H1 lift.", 3),
                (3, 2, "H2 added value was flat with softer incremental gains.", 5),
                (4, 2, "H2 underperformed H1 on added value by 3%.", 6),
            ]

            await conn.execute(
                metadata.tables["documents"].insert().values(
                    id=1,
                    filename="qbr_h1.pptx",
                    client_name="Acme",
                    period="H1",
                    status="enhanced",
                    executive_summary="H1 summary",
                )
            )
            await conn.execute(
                metadata.tables["documents"].insert().values(
                    id=2,
                    filename="qbr_h2.pptx",
                    client_name="Acme",
                    period="H2",
                    status="enhanced",
                    executive_summary="H2 summary",
                )
            )

            for chunk_id, doc_id, content, slide in chunks:
                embedding = await provider.embed_query(content)
                await conn.execute(
                    metadata.tables["chunks"].insert().values(
                        id=chunk_id,
                        document_id=doc_id,
                        content=content,
                        start_slide=slide,
                        end_slide=slide,
                        summary=None,
                        topics=None,
                        importance_score=None,
                        embedding=list(embedding.vector.values),
                        embedding_model=embedding.model,
                    )
                )
            try:
                await conn.execute(text("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"))
            except Exception:
                pass

        await engine.dispose()

    asyncio.run(_run())


def _validate_payload(payload: dict) -> str:
    for model in (
        OutputReady,
        OutputThinking,
        OutputUserMessage,
        OutputFinal,
        OutputError,
        OutputPing,
        OutputPong,
    ):
        try:
            model.model_validate(payload)
            return model.__name__
        except Exception:
            continue
    raise ValueError(f"Unrecognized payload: {payload}")


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "qbr_legacy_e2e.db"
    if db_path.exists():
        db_path.unlink()
    db_url = f"sqlite+aiosqlite:///{db_path.resolve()}"
    _setup_db(db_url)

    config = Config(
        output_protocol="websocket",
        use_stub_llm=True,
        database_url=db_url,
        embeddings_backend="hash",
        embeddings_model="ignored",
        storage_backend="sqlite",
        vector_backend="sqlite_embeddings",
        retrieval_top_k=3,
        rerank_top_n=4,
        retrieval_max_chunks_per_doc=1,
    )

    reranker = DebugReranker()
    planner = ToolRunnerPlanner()

    def orchestrator_factory(telemetry):
        engine = create_async_engine(db_url)
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        repository = SqlAlchemyKnowledgeRepository(sessionmaker=sessionmaker)
        vector_index = SqliteEmbeddingVectorIndex(repository=repository)
        infra = InfrastructureBundle(
            repository=repository,
            vector_index=vector_index,
            embeddings=HashEmbeddingsProvider(),
            reranker=reranker,
        )
        return AiAgentQbrOrchestrator(
            config,
            telemetry=telemetry,
            planner=planner,
            memory_store=InMemoryMemoryStore(max_turns=3, retrieval_turns=3),
            infrastructure=infra,
        )

    app = create_app(config=config, orchestrator_factory=orchestrator_factory)

    output_path = data_dir / "legacy_websocket_e2e.jsonl"
    summary_path = data_dir / "legacy_websocket_e2e_summary.txt"
    messages = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat/e2e-session") as ws:
            ws.send_json({"message": QUERY})
            for _ in range(40):
                payload = ws.receive_json()
                payload_type = _validate_payload(payload)
                messages.append({"type": payload_type, "payload": payload})
                if payload.get("status") in {"final", "error"}:
                    break

    with output_path.open("w", encoding="utf-8") as handle:
        for msg in messages:
            handle.write(json.dumps(msg) + "\n")

    context = ""
    if planner.last_llm_context:
        context = planner.last_llm_context.get("qbr_context", "") or ""

    search_summary = []
    if planner.last_search_results:
        for result in planner.last_search_results.results:
            search_summary.append(
                f"{result.title} | score={result.score:.3f} | {result.snippet}"
            )

    final_payload = next(
        (msg["payload"] for msg in messages if msg["payload"].get("status") == "final"),
        {},
    )
    final_text = final_payload.get("data", {}).get("content", "")

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Legacy WebSocket E2E Summary\n")
        handle.write(f"Query: {QUERY}\n")
        handle.write(f"Reranker calls: {reranker.calls}\n")
        handle.write("\nSearch Results (after rerank/MMR/doc caps):\n")
        for line in search_summary:
            handle.write(f"- {line}\n")
        handle.write("\nQBR Context:\n")
        handle.write(context or "None")
        handle.write("\n\nFinal Answer:\n")
        handle.write(final_text or "None")

    print(f"Wrote contract output to: {output_path}")
    print(f"Wrote summary to: {summary_path}")
    print(f"Reranker calls: {reranker.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
