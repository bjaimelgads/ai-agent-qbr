"""Application ports for QBR knowledge retrieval."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from qbr_agent.domain.entities import Chunk, Document
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector


@dataclass(frozen=True)
class EmbeddingResult:
    vector: EmbeddingVector
    model: str | None


@dataclass(frozen=True)
class VectorMatch:
    chunk_id: ChunkId
    score: float


@dataclass(frozen=True)
class TextMatch:
    chunk_id: ChunkId
    score: float


class KnowledgeRepository(Protocol):
    async def list_documents(
        self,
        *,
        limit: int = 50,
        client_name: str | None = None,
        status: str | None = None,
    ) -> list[Document]:
        ...

    async def fetch_chunks_by_ids(self, chunk_ids: Iterable[ChunkId]) -> list[Chunk]:
        ...

    async def fetch_chunk_embeddings(
        self,
        *,
        document_id: DocumentId | None = None,
        embedding_model: str | None = None,
    ) -> list[tuple[ChunkId, EmbeddingVector]]:
        ...

    async def search_chunks_text(
        self,
        *,
        query: str,
        limit: int,
        document_id: DocumentId | None = None,
    ) -> list[TextMatch]:
        ...


class VectorIndex(Protocol):
    async def search(
        self,
        *,
        vector: EmbeddingVector,
        top_k: int,
        document_id: DocumentId | None = None,
        embedding_model: str | None = None,
        min_score: float | None = None,
    ) -> list[VectorMatch]:
        ...


class EmbeddingsProvider(Protocol):
    async def embed_query(self, text: str) -> EmbeddingResult:
        ...
