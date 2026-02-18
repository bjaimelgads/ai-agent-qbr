"""Embedding providers for QBR retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from qbr_agent.application.ports import EmbeddingResult, EmbeddingsProvider
from qbr_agent.domain.value_objects import EmbeddingVector


@dataclass
class SentenceTransformersEmbeddingsProvider(EmbeddingsProvider):
    model_name: str
    normalize: bool = True

    _model: Any | None = None

    async def embed_query(self, text: str) -> EmbeddingResult:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vector = self._model.encode([text], normalize_embeddings=self.normalize)
        values = vector[0]
        if hasattr(values, "tolist"):
            values = values.tolist()
        return EmbeddingResult(
            vector=EmbeddingVector(tuple(float(value) for value in values)),
            model=f"sentence-transformers:{self.model_name}",
        )


@dataclass
class HashEmbeddingsProvider(EmbeddingsProvider):
    dimension: int = 256

    async def embed_query(self, text: str) -> EmbeddingResult:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for idx in range(self.dimension):
            byte = digest[idx % len(digest)]
            values.append((byte / 255.0) - 0.5)
        return EmbeddingResult(
            vector=EmbeddingVector(tuple(values)),
            model="hash:sha256",
        )
