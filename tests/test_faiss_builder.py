from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import JSON, Column, Integer, MetaData, String, Table, Text
from sqlalchemy.ext.asyncio import create_async_engine

faiss = pytest.importorskip("faiss")
pytest.importorskip("numpy")

from qbr_agent.infrastructure.faiss_builder import FaissBuildConfig, build_faiss_index


@pytest.mark.asyncio
async def test_build_faiss_index(tmp_path: Path) -> None:
    db_path = tmp_path / "qbr_faiss.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    metadata = MetaData()

    Table(
        "chunks",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("document_id", Integer),
        Column("content", Text),
        Column("embedding", JSON),
        Column("embedding_model", String(100)),
    )

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.execute(
            metadata.tables["chunks"].insert().values(
                id=1,
                document_id=1,
                content="Revenue grew",
                embedding=[0.1, 0.2, 0.3],
                embedding_model="test",
            )
        )

    config = FaissBuildConfig(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        base_dir=tmp_path / "faiss",
        normalize=True,
        index_type="Flat",
        metric="ip",
        embedding_model_filter=None,
        force=True,
    )
    built = await build_faiss_index(config)
    assert built is True
    assert (config.base_dir / "index.faiss").exists()
    assert (config.base_dir / "index_ids.json").exists()
