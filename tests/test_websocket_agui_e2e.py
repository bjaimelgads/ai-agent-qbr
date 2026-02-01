from __future__ import annotations

import asyncio
import os

import pytest
from fastapi.testclient import TestClient
from penguiflow.planner import PlannerFinish
from sqlalchemy import JSON, Column, Float, Integer, MetaData, String, Table, Text
from sqlalchemy.ext.asyncio import create_async_engine

from ai_agent_qbr.api.app import create_app
from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
from qbr_agent.infrastructure.embeddings import HashEmbeddingsProvider


class FakePlanner:
    async def run(self, *, query, llm_context, tool_context):
        answer = f"Answer: {query}\n\n{llm_context.get('qbr_context', '')}"
        return PlannerFinish(payload={"answer": answer}, metadata={})


def _setup_db(db_url: str) -> None:
    async def _run():
        engine = create_async_engine(db_url)
        metadata = MetaData()
        Table(
            "clients",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String(255), unique=True),
        )
        Table(
            "documents",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("filename", String(255)),
            Column("client_id", Integer),
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

        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

            provider = HashEmbeddingsProvider()
            embedding = await provider.embed_query("Revenue grew by 10%")
            await conn.execute(
                metadata.tables["clients"].insert().values(
                    id=1,
                    name="Acme",
                )
            )
            await conn.execute(
                metadata.tables["documents"].insert().values(
                    id=1,
                    filename="qbr.pptx",
                    client_id=1,
                    period="Q1",
                    status="enhanced",
                    executive_summary="Summary",
                )
            )
            await conn.execute(
                metadata.tables["chunks"].insert().values(
                    id=1,
                    document_id=1,
                    content="Revenue grew by 10%",
                    start_slide=2,
                    end_slide=2,
                    summary=None,
                    topics=None,
                    importance_score=None,
                    embedding=list(embedding.vector.values),
                    embedding_model=embedding.model,
                )
            )

        await engine.dispose()

    asyncio.run(_run())


@pytest.mark.skipif(os.getenv("RUN_E2E") != "1", reason="Set RUN_E2E=1 to run WebSocket e2e test.")
def test_agui_websocket_streams_context(tmp_path):
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'qbr_e2e.db'}"
    _setup_db(db_url)

    use_stub_llm = os.getenv("USE_STUB_LLM", "true").lower() in {"1", "true", "yes", "on"}
    config = Config(
        output_protocol="agui",
        use_stub_llm=use_stub_llm,
        database_url=db_url,
        embeddings_backend="hash",
        embeddings_model="ignored",
        storage_backend="sqlite",
        vector_backend="sqlite_embeddings",
        rerank_backend="none",
    )

    def orchestrator_factory(telemetry):
        return AiAgentQbrOrchestrator(
            config,
            telemetry=telemetry,
            planner=FakePlanner(),
            memory_store=InMemoryMemoryStore(max_turns=3, retrieval_turns=3),
        )

    app = create_app(config=config, orchestrator_factory=orchestrator_factory)

    messages = []
    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat/test-session") as ws:
            ws.send_json({"message": "What happened to revenue?"})
            for _ in range(20):
                payload = ws.receive_json()
                messages.append(payload)
                if payload.get("type") == "RUN_FINISHED":
                    break

    assert any(msg.get("type") == "RUN_STARTED" for msg in messages)
    assert any(msg.get("type") == "TEXT_MESSAGE_CONTENT" for msg in messages)
    assert any("Revenue grew by 10%" in msg.get("delta", "") for msg in messages)
