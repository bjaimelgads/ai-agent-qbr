from __future__ import annotations

import json
from pathlib import Path

import pytest

faiss = pytest.importorskip("faiss")
np = pytest.importorskip("numpy")

from qbr_agent.domain.value_objects import EmbeddingVector
from qbr_agent.infrastructure.faiss_index import FaissVectorIndex


def _write_index(base_dir: Path) -> None:
    vectors = np.asarray(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype="float32",
    )
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(base_dir / "index.faiss"))
    with (base_dir / "index_ids.json").open("w", encoding="utf-8") as handle:
        json.dump({"ids": [11, 22]}, handle)


@pytest.mark.asyncio
async def test_faiss_index_search(tmp_path: Path) -> None:
    _write_index(tmp_path)
    index = FaissVectorIndex(base_dir=tmp_path, normalize=True)
    matches = await index.search(vector=EmbeddingVector((0.1, 0.0, 0.0)), top_k=1)
    assert matches
    assert matches[0].chunk_id.value == 11
