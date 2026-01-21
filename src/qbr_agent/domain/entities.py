"""Domain entities for QBR knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .value_objects import ChunkId, DocumentId, EmbeddingVector, Score


@dataclass(frozen=True)
class Document:
    document_id: DocumentId
    filename: str | None
    file_path: str | None
    client_name: str | None
    period: str | None
    status: str | None
    executive_summary: str | None


@dataclass(frozen=True)
class Slide:
    slide_id: int
    document_id: DocumentId
    slide_number: int
    title: str | None
    content: str | None


@dataclass(frozen=True)
class Embedding:
    vector: EmbeddingVector
    model: str | None


@dataclass(frozen=True)
class Chunk:
    chunk_id: ChunkId
    document_id: DocumentId
    content: str
    start_slide: int | None
    end_slide: int | None
    summary: str | None
    topics: list[str] | None
    importance_score: float | None
    embedding: Embedding | None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: Score


@dataclass(frozen=True)
class Answer:
    question: str
    text: str | None
    context: str
    citations: list[RetrievalResult]
