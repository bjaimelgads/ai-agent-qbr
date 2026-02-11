"""Unit tests for search tool."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_agent_qbr.models import Query, SearchResults
from ai_agent_qbr.tools.search import search_documents
from qbr_agent.application.ports import (
    EmbeddingResult,
    EmbeddingsProvider,
    KnowledgeRepository,
    TextMatch,
    VectorIndex,
    VectorMatch,
)
from qbr_agent.application.use_cases import HybridSearchKnowledge
from qbr_agent.domain.entities import Chunk, Document, Embedding
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector


@dataclass
class FakeRepository(KnowledgeRepository):
    chunk: Chunk

    async def list_documents(self, *, limit=50, client_name=None, status=None):
        return []

    async def fetch_documents_by_ids(self, document_ids):
        if any(doc_id.value == self.chunk.document_id.value for doc_id in document_ids):
            return [
                Document(
                    document_id=self.chunk.document_id,
                    filename="qbr_test.pptx",
                    file_path="/tmp/qbr_test.pptx",
                    client_name=None,
                    period=None,
                    status=None,
                    executive_summary=None,
                )
            ]
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


class DummyToolContext:
    def __init__(self, search_use_case: HybridSearchKnowledge):
        self.tool_context = {
            "tenant_id": "test-tenant",
            "qbr_search_use_case": search_use_case,
            "output_protocol": "websocket",
        }


@pytest.mark.asyncio
async def test_search_documents_returns_results() -> None:
    chunk = Chunk(
        chunk_id=ChunkId(1),
        document_id=DocumentId(1),
        content="QBR content",
        start_slide=1,
        end_slide=1,
        summary=None,
        topics=None,
        importance_score=None,
        embedding=Embedding(vector=EmbeddingVector((0.1, 0.2, 0.3)), model="test"),
    )
    use_case = HybridSearchKnowledge(
        repository=FakeRepository(chunk),
        vector_index=FakeVectorIndex(),
        embeddings=FakeEmbeddingsProvider(),
    )
    dummy_ctx = DummyToolContext(use_case)

    result = await search_documents(Query(question="What is in the QBR?"), dummy_ctx)

    assert isinstance(result, SearchResults)
    assert result.results
    assert result.results[0].title
    assert result.results[0].snippet


@pytest.mark.asyncio
async def test_search_documents_with_different_query() -> None:
    chunk = Chunk(
        chunk_id=ChunkId(1),
        document_id=DocumentId(1),
        content="Another QBR content",
        start_slide=None,
        end_slide=None,
        summary=None,
        topics=None,
        importance_score=None,
        embedding=Embedding(vector=EmbeddingVector((0.1, 0.2, 0.3)), model="test"),
    )
    use_case = HybridSearchKnowledge(
        repository=FakeRepository(chunk),
        vector_index=FakeVectorIndex(),
        embeddings=FakeEmbeddingsProvider(),
    )
    dummy_ctx = DummyToolContext(use_case)

    result = await search_documents(Query(question="How do I use tools?"), dummy_ctx)

    assert isinstance(result, SearchResults)
    assert result.results
