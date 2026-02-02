from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
from qbr_agent.infrastructure.embeddings import HashEmbeddingsProvider
from qbr_agent.infrastructure.factory import InfrastructureBundle
from qbr_agent.infrastructure.reranker import NoopReranker
from qbr_agent.infrastructure.sqlalchemy_repository import SqlAlchemyKnowledgeRepository
from qbr_agent.infrastructure.vector_index import SqliteEmbeddingVectorIndex


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
    )

    response = await orchestrator.execute(
        "CPA for Nike in Q2 2025 in US",
        tenant_id="test",
        user_id="user",
        session_id="session",
    )

    assert response.answer is not None
    assert "Sources:" in response.answer
