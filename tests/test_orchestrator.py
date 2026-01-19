"""Integration tests for the AiAgentQbrOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_agent_qbr.config import Config
from ai_agent_qbr.infrastructure.memory_store import InMemoryMemoryStore
from ai_agent_qbr.orchestrator import AiAgentQbrOrchestrator
from qbr_agent.application.ports import (
    EmbeddingResult,
    EmbeddingsProvider,
    KnowledgeRepository,
    TextMatch,
    VectorIndex,
    VectorMatch,
)
from qbr_agent.domain.entities import Chunk, Embedding
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector
from qbr_agent.infrastructure.factory import InfrastructureBundle


@dataclass
class FakeRepository(KnowledgeRepository):
    chunk: Chunk

    async def list_documents(self, *, limit=50, client_name=None, status=None):
        return []

    async def fetch_chunks_by_ids(self, chunk_ids):
        if any(cid.value == self.chunk.chunk_id.value for cid in chunk_ids):
            return [self.chunk]
        return []

    async def fetch_chunk_embeddings(self, *, document_id=None, embedding_model=None):
        return [(self.chunk.chunk_id, self.chunk.embedding.vector)]

    async def search_chunks_text(self, *, query, limit, document_id=None):
        del query, limit, document_id
        return [TextMatch(chunk_id=self.chunk.chunk_id, score=1.0)]


@dataclass
class FakeVectorIndex(VectorIndex):
    async def search(self, *, vector, top_k, document_id=None, embedding_model=None, min_score=None):
        return [VectorMatch(chunk_id=ChunkId(1), score=0.9)]


@dataclass
class FakeEmbeddingsProvider(EmbeddingsProvider):
    async def embed_query(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=EmbeddingVector((0.1, 0.2, 0.3)), model="test")


@pytest.mark.asyncio
async def test_execute_returns_agent_response() -> None:
    config = Config()
    chunk = Chunk(
        chunk_id=ChunkId(1),
        document_id=DocumentId(1),
        content="QBR summary content",
        start_slide=1,
        end_slide=1,
        summary=None,
        topics=None,
        importance_score=None,
        embedding=Embedding(
            vector=EmbeddingVector((0.1, 0.2, 0.3)),
            model="test",
        ),
    )
    infra = InfrastructureBundle(
        repository=FakeRepository(chunk),
        vector_index=FakeVectorIndex(),
        embeddings=FakeEmbeddingsProvider(),
    )
    orchestrator = AiAgentQbrOrchestrator(
        config,
        memory_store=InMemoryMemoryStore(max_turns=3, retrieval_turns=3),
        infrastructure=infra,
    )

    response = await orchestrator.execute(
        query="Tell me about the QBR",
        tenant_id="tenant-a",
        user_id="user-a",
        session_id="session-a",
    )

    assert response.answer
    assert response.trace_id
    await orchestrator.stop()


@pytest.mark.asyncio
async def test_stop_marks_orchestrator_inactive() -> None:
    orchestrator = AiAgentQbrOrchestrator(Config())
    assert orchestrator._started is True  # type: ignore[attr-defined]
    await orchestrator.stop()
    assert orchestrator._started is False  # type: ignore[attr-defined]
