from __future__ import annotations

from dataclasses import dataclass

import pytest

from qbr_agent.application.ports import (
    EmbeddingResult,
    EmbeddingsProvider,
    KnowledgeRepository,
    TextMatch,
    VectorIndex,
    VectorMatch,
)
from qbr_agent.application.use_cases import AnswerQuestion
from qbr_agent.domain.entities import Chunk, Embedding
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector


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
        return [VectorMatch(chunk_id=ChunkId(1), score=0.8)]


@dataclass
class FakeEmbeddingsProvider(EmbeddingsProvider):
    async def embed_query(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=EmbeddingVector((0.1, 0.2, 0.3)), model="test")


@pytest.mark.asyncio
async def test_answer_question_builds_context() -> None:
    chunk = Chunk(
        chunk_id=ChunkId(1),
        document_id=DocumentId(7),
        content="Revenue grew by 10%",
        start_slide=2,
        end_slide=3,
        summary=None,
        topics=None,
        importance_score=None,
        embedding=Embedding(vector=EmbeddingVector((0.1, 0.2, 0.3)), model="test"),
    )
    use_case = AnswerQuestion(
        repository=FakeRepository(chunk),
        vector_index=FakeVectorIndex(),
        embeddings=FakeEmbeddingsProvider(),
    )

    answer = await use_case.execute(query="What happened to revenue?")

    assert "Revenue grew" in answer.context
    assert "Doc 7" in answer.context
