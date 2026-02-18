"""FAISS-backed vector index for QBR retrieval."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qbr_agent.application.ports import VectorIndex, VectorMatch
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled in factory
    raise RuntimeError("numpy is required for FAISS retrieval") from exc

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - handled in factory
    raise RuntimeError("faiss is required for FAISS retrieval") from exc


_LOGGER = logging.getLogger(__name__)


@dataclass
class FaissVectorIndex(VectorIndex):
    base_dir: Path
    normalize: bool = True

    @property
    def index_path(self) -> Path:
        return self.base_dir / "index.faiss"

    @property
    def ids_path(self) -> Path:
        return self.base_dir / "index_ids.json"

    async def search(
        self,
        *,
        vector: EmbeddingVector,
        top_k: int,
        document_id: DocumentId | None = None,
        embedding_model: str | None = None,
        min_score: float | None = None,
    ) -> list[VectorMatch]:
        del document_id, embedding_model
        if top_k <= 0:
            return []
        if not self.index_path.exists() or not self.ids_path.exists():
            _LOGGER.warning("FAISS index missing; run build_faiss_index.py")
            return []

        index = faiss.read_index(str(self.index_path))
        ids = _load_ids(self.ids_path)
        if index.ntotal == 0:
            return []

        query = np.asarray([list(vector.values)], dtype="float32")
        if self.normalize:
            faiss.normalize_L2(query)
        distances, indices = index.search(query, min(top_k, index.ntotal))

        results: list[VectorMatch] = []
        for raw_id, distance in zip(indices[0], distances[0], strict=False):
            if raw_id < 0 or raw_id >= len(ids):
                continue
            score = _distance_to_score(float(distance))
            if min_score is not None and score < min_score:
                continue
            results.append(VectorMatch(chunk_id=ChunkId(ids[raw_id]), score=score))
        return results


def _load_ids(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    ids = payload.get("ids", [])
    return [int(value) for value in ids]


def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))
