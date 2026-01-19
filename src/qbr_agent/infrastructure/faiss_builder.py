"""Helpers for building FAISS indices from stored embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from qbr_agent.infrastructure.sqlalchemy_repository import SqlAlchemyKnowledgeRepository


@dataclass(frozen=True)
class FaissBuildConfig:
    database_url: str
    base_dir: Path
    normalize: bool = True
    index_type: str = "Flat"
    metric: str = "ip"
    embedding_model_filter: str | None = None
    force: bool = False


async def build_faiss_index(config: FaissBuildConfig) -> bool:
    """Build FAISS index from stored embeddings.

    Returns True if a new index was built.
    """
    index_path = config.base_dir / "index.faiss"
    ids_path = config.base_dir / "index_ids.json"
    if not config.force and index_path.exists() and ids_path.exists():
        return False

    try:  # pragma: no cover - optional dependency
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - handled by caller
        raise RuntimeError("numpy is required to build FAISS indices") from exc

    try:  # pragma: no cover - optional dependency
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - handled by caller
        raise RuntimeError("faiss is required to build FAISS indices") from exc

    config.base_dir.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(config.database_url)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyKnowledgeRepository(sessionmaker=sessionmaker)

    embeddings = await repository.fetch_chunk_embeddings(
        embedding_model=config.embedding_model_filter
    )
    if not embeddings:
        await engine.dispose()
        return False

    ids = [chunk_id.value for chunk_id, _ in embeddings]
    vectors = [list(vector.values) for _, vector in embeddings]

    matrix = np.asarray(vectors, dtype="float32")
    if config.normalize:
        faiss.normalize_L2(matrix)

    dimension = matrix.shape[1]
    index_type = config.index_type.lower()
    metric = config.metric.lower()

    if index_type != "flat":
        raise ValueError(f"Unsupported FAISS_INDEX_TYPE: {config.index_type}")

    if metric == "ip":
        index = faiss.IndexFlatIP(dimension)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported FAISS_METRIC: {config.metric}")

    index.add(matrix)

    faiss.write_index(index, str(index_path))
    with ids_path.open("w", encoding="utf-8") as handle:
        json.dump({"ids": ids}, handle)

    await engine.dispose()
    return True
