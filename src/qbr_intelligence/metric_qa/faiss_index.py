"""FAISS-backed vector index for metric fact embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetricFactFaissIndex:
    base_dir: Path
    normalize: bool = True
    metric: str = "ip"
    embedding_model: str | None = None
    _index: Any | None = field(default=None, init=False, repr=False)
    _ids: list[int] | None = field(default=None, init=False, repr=False)
    _meta: dict[str, Any] | None = field(default=None, init=False, repr=False)

    @property
    def index_path(self) -> Path:
        return self.base_dir / "index.faiss"

    @property
    def ids_path(self) -> Path:
        return self.base_dir / "index_ids.json"

    def _load(self) -> bool:
        if self._index is not None and self._ids is not None:
            return True
        if not self.index_path.exists() or not self.ids_path.exists():
            return False
        try:  # pragma: no cover - optional dependency
            import faiss  # type: ignore
        except ModuleNotFoundError:
            return False
        self._index = faiss.read_index(str(self.index_path))
        with self.ids_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._ids = [int(value) for value in payload.get("ids", [])]
        self._meta = payload
        return True

    def search(
        self,
        *,
        vector: list[float] | tuple[float, ...],
        top_k: int,
        embedding_model: str | None = None,
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        if not self._load() or self._index is None or self._ids is None:
            return []
        if embedding_model and self._meta:
            meta_model = self._meta.get("embedding_model")
            if meta_model and meta_model != embedding_model:
                return []
        try:  # pragma: no cover - optional dependency
            import numpy as np  # type: ignore
        except ModuleNotFoundError:
            return []
        try:  # pragma: no cover - optional dependency
            import faiss  # type: ignore
        except ModuleNotFoundError:
            return []

        query = np.asarray([list(vector)], dtype="float32")
        if self.normalize:
            faiss.normalize_L2(query)
        distances, indices = self._index.search(query, min(top_k, self._index.ntotal))

        results: list[tuple[int, float]] = []
        for raw_id, distance in zip(indices[0], distances[0], strict=False):
            if raw_id < 0 or raw_id >= len(self._ids):
                continue
            score = _distance_to_score(float(distance), metric=self.metric)
            results.append((self._ids[raw_id], score))
        return results

    @classmethod
    def from_env(cls) -> "MetricFactFaissIndex":
        import os

        base_dir = Path(os.getenv("METRIC_FAISS_DIR", "./data/faiss_metric"))
        normalize = os.getenv("METRIC_FAISS_NORMALIZE", "true").lower() in {"1", "true", "yes"}
        metric = os.getenv("METRIC_FAISS_METRIC", "ip").lower()
        embedding_model = os.getenv("METRIC_FAISS_EMBEDDING_MODEL_FILTER", "").strip() or None
        return cls(
            base_dir=base_dir,
            normalize=normalize,
            metric=metric,
            embedding_model=embedding_model,
        )


def _distance_to_score(distance: float, *, metric: str) -> float:
    if metric == "ip":
        return distance
    return 1.0 / (1.0 + max(distance, 0.0))
