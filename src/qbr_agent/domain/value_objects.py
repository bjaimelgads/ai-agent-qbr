"""Domain value objects for QBR knowledge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentId:
    value: int


@dataclass(frozen=True)
class ChunkId:
    value: int


@dataclass(frozen=True)
class EmbeddingVector:
    values: tuple[float, ...]


@dataclass(frozen=True)
class Score:
    value: float
