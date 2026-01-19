"""Build a FAISS index from stored QBR chunk embeddings."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from qbr_agent.infrastructure.faiss_builder import FaissBuildConfig, build_faiss_index


async def main() -> None:
    config = FaissBuildConfig(
        database_url=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///qbr_intelligence.db"),
        base_dir=Path(os.getenv("FAISS_DIR", "./data/faiss")),
        normalize=os.getenv("FAISS_NORMALIZE", "true").lower() in {"1", "true", "yes"},
        index_type=os.getenv("FAISS_INDEX_TYPE", "Flat"),
        metric=os.getenv("FAISS_METRIC", "ip"),
        embedding_model_filter=os.getenv("FAISS_EMBEDDING_MODEL_FILTER", "").strip() or None,
        force=os.getenv("FAISS_FORCE_REBUILD", "").lower() in {"1", "true", "yes"},
    )

    built = await build_faiss_index(config)
    if built:
        print(f"FAISS index built at {config.base_dir}")
    else:
        print("FAISS index already exists or no embeddings found.")


if __name__ == "__main__":
    asyncio.run(main())
