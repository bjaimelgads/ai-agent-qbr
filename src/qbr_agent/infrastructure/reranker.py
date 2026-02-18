"""Reranker implementations for QBR retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from qbr_agent.application.ports import Reranker
from qbr_agent.domain.entities import Chunk

_LOGGER = logging.getLogger(__name__)


@dataclass
class NoopReranker(Reranker):
    """Fallback reranker that preserves existing ordering."""

    async def score(self, *, query: str, chunks: Sequence[Chunk]) -> list[float]:
        del query
        return [0.0 for _ in chunks]


@dataclass
class CrossEncoderReranker(Reranker):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_length: int | None = None
    _model: Any | None = None
    _ready: bool = False

    def _ensure_model(self) -> Any | None:
        if self._ready:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("CrossEncoder reranker unavailable: %s", exc)
            self._ready = True
            self._model = None
            return None
        kwargs: dict[str, Any] = {}
        if self.max_length:
            kwargs["max_length"] = self.max_length
        self._model = CrossEncoder(self.model_name, **kwargs)
        self._ready = True
        return self._model

    async def score(self, *, query: str, chunks: Sequence[Chunk]) -> list[float]:
        model = self._ensure_model()
        if not model or not chunks:
            return [0.0 for _ in chunks]
        pairs = [(query, chunk.content) for chunk in chunks]
        try:
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("CrossEncoder rerank failed: %s", exc)
            return [0.0 for _ in chunks]
        return [float(score) for score in scores]
