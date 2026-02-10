from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from qbr_intelligence.db.migrate import migrate_sqlite_to_postgres
from qbr_intelligence.db.models import Base, Chunk, Document


@pytest.mark.e2e
def test_sqlite_to_postgres_migration(tmp_path) -> None:
    pg_url = os.getenv("QBR_PG_TEST_URL")
    if not pg_url:
        pytest.skip("QBR_PG_TEST_URL not set; skipping Postgres migration e2e test")

    sqlite_path = tmp_path / "qbr_test.db"
    sqlite_url = f"sqlite:///{sqlite_path}"

    src_engine = create_engine(sqlite_url)
    Base.metadata.create_all(src_engine)
    with Session(src_engine) as session:
        doc = Document(
            id=1,
            filename="qbr.pptx",
            mime_type="application/vnd.ms-powerpoint",
            status="enhanced",
            executive_summary="Summary",
        )
        chunk = Chunk(
            id=1,
            document_id=1,
            content="Revenue grew by 10%",
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test",
        )
        session.add_all([doc, chunk])
        session.commit()

    result = migrate_sqlite_to_postgres(
        sqlite_url=sqlite_url,
        postgres_url=pg_url,
        truncate=True,
        create_tables=True,
        reset_sequences=True,
        create_fts_indexes=False,
    )
    assert result.table_counts.get("documents", 0) >= 1
    assert result.table_counts.get("chunks", 0) >= 1

    dst_engine = create_engine(pg_url)
    with Session(dst_engine) as session:
        docs = session.execute(select(Document)).scalars().all()
        chunks = session.execute(select(Chunk)).scalars().all()
    assert any(doc.filename == "qbr.pptx" for doc in docs)
    assert any(chunk.content.startswith("Revenue") for chunk in chunks)
