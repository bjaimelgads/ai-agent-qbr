from __future__ import annotations

from dataclasses import dataclass

import pytest

from qbr_agent.application.ports import (
    EmbeddingResult,
    EmbeddingsProvider,
    KnowledgeRepository,
    Reranker,
    TextMatch,
    VectorIndex,
    VectorMatch,
)
from qbr_agent.application.use_cases import AnswerQuestion, HybridSearchKnowledge
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
        return [VectorMatch(chunk_id=ChunkId(1), score=0.8)]


@dataclass
class FakeEmbeddingsProvider(EmbeddingsProvider):
    async def embed_query(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(vector=EmbeddingVector((0.1, 0.2, 0.3)), model="test")


@dataclass
class FakeReranker(Reranker):
    scores: list[float]

    async def score(self, *, query: str, chunks) -> list[float]:
        del query
        return self.scores[: len(chunks)]


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
    assert "File: qbr_test.pptx" in answer.context


@pytest.mark.asyncio
async def test_hybrid_search_applies_rerank_and_doc_cap() -> None:
    chunks = [
        Chunk(
            chunk_id=ChunkId(1),
            document_id=DocumentId(10),
            content="Doc10 chunk 1",
            start_slide=None,
            end_slide=None,
            summary=None,
            topics=None,
            importance_score=None,
            embedding=None,
        ),
        Chunk(
            chunk_id=ChunkId(2),
            document_id=DocumentId(10),
            content="Doc10 chunk 2",
            start_slide=None,
            end_slide=None,
            summary=None,
            topics=None,
            importance_score=None,
            embedding=None,
        ),
        Chunk(
            chunk_id=ChunkId(3),
            document_id=DocumentId(10),
            content="Doc10 chunk 3",
            start_slide=None,
            end_slide=None,
            summary=None,
            topics=None,
            importance_score=None,
            embedding=None,
        ),
        Chunk(
            chunk_id=ChunkId(4),
            document_id=DocumentId(20),
            content="Doc20 chunk 1",
            start_slide=None,
            end_slide=None,
            summary=None,
            topics=None,
            importance_score=None,
            embedding=None,
        ),
    ]

    @dataclass
    class MultiChunkRepository(KnowledgeRepository):
        all_chunks: list[Chunk]

        async def list_documents(self, *, limit=50, client_name=None, status=None):
            return []

        async def fetch_documents_by_ids(self, document_ids):
            return []

        async def fetch_chunks_by_ids(self, chunk_ids):
            wanted = {cid.value for cid in chunk_ids}
            return [chunk for chunk in self.all_chunks if chunk.chunk_id.value in wanted]

        async def fetch_chunk_embeddings(self, *, document_id=None, embedding_model=None):
            return []

        async def search_chunks_text(self, *, query, limit, document_id=None):
            del query, limit, document_id
            return [
                TextMatch(chunk_id=ChunkId(1), score=1.0),
                TextMatch(chunk_id=ChunkId(2), score=0.9),
                TextMatch(chunk_id=ChunkId(3), score=0.8),
                TextMatch(chunk_id=ChunkId(4), score=0.7),
            ]

    @dataclass
    class MultiVectorIndex(VectorIndex):
        async def search(self, *, vector, top_k, document_id=None, embedding_model=None, min_score=None):
            del vector, top_k, document_id, embedding_model, min_score
            return [
                VectorMatch(chunk_id=ChunkId(1), score=0.95),
                VectorMatch(chunk_id=ChunkId(2), score=0.9),
                VectorMatch(chunk_id=ChunkId(3), score=0.85),
                VectorMatch(chunk_id=ChunkId(4), score=0.8),
            ]

    use_case = HybridSearchKnowledge(
        repository=MultiChunkRepository(chunks),
        vector_index=MultiVectorIndex(),
        embeddings=FakeEmbeddingsProvider(),
        reranker=FakeReranker([0.1, 0.2, 0.9, 0.8]),
        rerank_top_n=4,
        max_chunks_per_doc=2,
    )

    results = await use_case.execute(query="test rerank", top_k=3)

    assert [result.chunk.chunk_id.value for result in results] == [3, 4, 2]
