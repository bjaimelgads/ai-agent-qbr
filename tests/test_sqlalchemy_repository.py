from __future__ import annotations

import pytest
from sqlalchemy import JSON, Column, Float, Integer, MetaData, String, Table, Text, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from qbr_agent.domain.value_objects import DocumentId, EmbeddingVector
from qbr_agent.infrastructure.sqlalchemy_repository import SqlAlchemyKnowledgeRepository
from qbr_agent.infrastructure.vector_index import SqliteEmbeddingVectorIndex


@pytest.mark.asyncio
async def test_repository_and_vector_index(tmp_path) -> None:
    db_path = tmp_path / "qbr_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    metadata = MetaData()

    Table(
        "documents",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("filename", String(255)),
        Column("client_name", String(255)),
        Column("period", String(100)),
        Column("status", String(50)),
        Column("executive_summary", Text),
    )
    Table(
        "chunks",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("document_id", Integer),
        Column("content", Text),
        Column("start_slide", Integer),
        Column("end_slide", Integer),
        Column("summary", Text),
        Column("topics", JSON),
        Column("importance_score", Float),
        Column("embedding", JSON),
        Column("embedding_model", String(100)),
    )

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.execute(
            metadata.tables["documents"].insert().values(
                id=1,
                filename="qbr.pptx",
                client_name="Acme",
                period="Q1",
                status="enhanced",
                executive_summary="Summary",
            )
        )
        await conn.execute(
            metadata.tables["chunks"].insert().values(
                id=1,
                document_id=1,
                content="Revenue grew by 10%",
                start_slide=2,
                end_slide=2,
                summary=None,
                topics=None,
                importance_score=None,
                embedding=[0.1, 0.2, 0.3],
                embedding_model="test",
            )
        )

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyKnowledgeRepository(sessionmaker=sessionmaker)
    vector_index = SqliteEmbeddingVectorIndex(repository=repository)

    matches = await vector_index.search(
        vector=EmbeddingVector((0.1, 0.2, 0.3)),
        top_k=1,
        embedding_model="test",
    )
    assert matches
    assert matches[0].chunk_id.value == 1

    chunks = await repository.fetch_chunks_by_ids([matches[0].chunk_id])
    assert chunks
    assert chunks[0].content.startswith("Revenue")

    text_hits = await repository.search_chunks_text(
        query="Revenue",
        limit=5,
        document_id=DocumentId(1),
    )
    assert text_hits


@pytest.mark.asyncio
async def test_repository_fts_bm25_ranking(tmp_path) -> None:
    db_path = tmp_path / "qbr_fts_test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    metadata = MetaData()

    Table(
        "documents",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("filename", String(255)),
        Column("client_name", String(255)),
        Column("period", String(100)),
        Column("status", String(50)),
        Column("executive_summary", Text),
    )
    Table(
        "chunks",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("document_id", Integer),
        Column("content", Text),
        Column("start_slide", Integer),
        Column("end_slide", Integer),
        Column("summary", Text),
        Column("topics", JSON),
        Column("importance_score", Float),
        Column("embedding", JSON),
        Column("embedding_model", String(100)),
    )

    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        try:
            await conn.execute(
                text(
                    "CREATE VIRTUAL TABLE chunks_fts "
                    "USING fts5(content, content='chunks', content_rowid='id')"
                )
            )
        except OperationalError as exc:
            await engine.dispose()
            pytest.skip(f"FTS5 not available: {exc}")

        await conn.execute(
            metadata.tables["chunks"].insert(),
            [
                {"id": 1, "document_id": 1, "content": "alpha beta gamma"},
                {"id": 2, "document_id": 1, "content": "alpha"},
                {"id": 3, "document_id": 1, "content": "beta beta beta"},
            ],
        )
        await conn.execute(text("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"))

    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repository = SqlAlchemyKnowledgeRepository(
        sessionmaker=sessionmaker,
        text_search_backend="fts5",
    )

    text_hits = await repository.search_chunks_text(
        query="alpha beta",
        limit=5,
        document_id=DocumentId(1),
    )
    assert text_hits
    assert text_hits[0].chunk_id.value == 1
    assert isinstance(text_hits[0].score, float)
