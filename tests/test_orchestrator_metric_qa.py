from __future__ import annotations

import pytest
from penguiflow.planner import PlannerFinish
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator, _format_metric_answer
from qbr_agent.infrastructure.embeddings import HashEmbeddingsProvider
from qbr_agent.infrastructure.factory import InfrastructureBundle
from qbr_agent.infrastructure.reranker import NoopReranker
from qbr_agent.infrastructure.sqlalchemy_repository import SqlAlchemyKnowledgeRepository
from qbr_agent.infrastructure.vector_index import SqliteEmbeddingVectorIndex
from qbr_intelligence.metric_qa import MetricQueryEngine


class MetricPlanner:
    def __init__(self, database_url: str) -> None:
        self._engine = MetricQueryEngine(database_url=database_url)

    async def run(self, *, query, llm_context, tool_context):
        result = await self._engine.query(query)
        payload = result.answer.model_dump()
        payload["raw_answer"] = _format_metric_answer(result.answer)
        return PlannerFinish(reason="answer_complete", payload=payload, metadata={})


@pytest.mark.asyncio
async def test_orchestrator_routes_metric_queries(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    config = Config(
        output_protocol="legacy",
        use_stub_llm=True,
        database_url=metric_db,
        embeddings_backend="hash",
        embeddings_model="ignored",
        storage_backend="sqlite",
        vector_backend="sqlite_embeddings",
        metric_router_enabled=True,
        guardrails_enabled=False,
    )

    engine = create_async_engine(metric_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyKnowledgeRepository(sessionmaker=sessionmaker)
    vector_index = SqliteEmbeddingVectorIndex(repository=repository)
    infra = InfrastructureBundle(
        repository=repository,
        vector_index=vector_index,
        embeddings=HashEmbeddingsProvider(),
        reranker=NoopReranker(),
    )

    orchestrator = AiAgentQbrOrchestrator(
        config,
        memory_store=InMemoryMemoryStore(max_turns=3, retrieval_turns=3),
        infrastructure=infra,
        planner=MetricPlanner(metric_db),
    )

    response = await orchestrator.execute(
        "CPA for Nike in Q2 2025 in US",
        tenant_id="test",
        user_id="user",
        session_id="session",
    )

    assert response.answer is not None
    assert "Sources:" in response.answer


@pytest.mark.asyncio
async def test_orchestrator_returns_metric_grid_for_multiple_values(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    config = Config(
        output_protocol="legacy",
        use_stub_llm=True,
        database_url=metric_db,
        embeddings_backend="hash",
        embeddings_model="ignored",
        storage_backend="sqlite",
        vector_backend="sqlite_embeddings",
        metric_router_enabled=True,
        guardrails_enabled=False,
    )

    engine = create_async_engine(metric_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyKnowledgeRepository(sessionmaker=sessionmaker)
    vector_index = SqliteEmbeddingVectorIndex(repository=repository)
    infra = InfrastructureBundle(
        repository=repository,
        vector_index=vector_index,
        embeddings=HashEmbeddingsProvider(),
        reranker=NoopReranker(),
    )

    orchestrator = AiAgentQbrOrchestrator(
        config,
        memory_store=InMemoryMemoryStore(max_turns=3, retrieval_turns=3),
        infrastructure=infra,
        planner=MetricPlanner(metric_db),
    )

    response = await orchestrator.execute(
        "Compare CPA for Nike in US",
        tenant_id="test",
        user_id="user",
        session_id="session",
    )

    assert response.artifacts is not None
    assert response.artifacts.get("type") == "datagrid"
    rows = response.artifacts.get("rows") or []
    assert len(rows) > 1
    for row in rows:
        assert "metric" in row
        assert "value" in row
        assert "period" in row
        assert "region" in row
        assert "client" in row
        assert "metric_url" in row
