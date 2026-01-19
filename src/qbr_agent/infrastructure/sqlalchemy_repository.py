"""SQLAlchemy repository for QBR knowledge."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import logging
from typing import Any

from sqlalchemy import MetaData, Table, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from qbr_agent.application.ports import KnowledgeRepository, TextMatch
from qbr_agent.domain.entities import Chunk, Document, Embedding
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector

_METADATA = MetaData()

_DOCUMENTS: Table | None = None
_CHUNKS: Table | None = None
_WARNED_MISSING_TABLES = False

_LOGGER = logging.getLogger(__name__)


@dataclass
class SqlAlchemyKnowledgeRepository(KnowledgeRepository):
    sessionmaker: async_sessionmaker[AsyncSession]

    async def _ensure_reflection(self, session: AsyncSession) -> None:
        global _DOCUMENTS, _CHUNKS, _WARNED_MISSING_TABLES
        if _DOCUMENTS is not None and _CHUNKS is not None:
            return

        conn = await session.connection()
        await conn.run_sync(_METADATA.reflect)
        _DOCUMENTS = _METADATA.tables.get("documents")
        _CHUNKS = _METADATA.tables.get("chunks")
        if not _WARNED_MISSING_TABLES and (_DOCUMENTS is None or _CHUNKS is None):
            missing = [
                name
                for name, table in (("documents", _DOCUMENTS), ("chunks", _CHUNKS))
                if table is None
            ]
            _LOGGER.warning("Missing tables in QBR database: %s", ", ".join(missing))
            _WARNED_MISSING_TABLES = True

    async def list_documents(
        self,
        *,
        limit: int = 50,
        client_name: str | None = None,
        status: str | None = None,
    ) -> list[Document]:
        async with self.sessionmaker() as session:
            await self._ensure_reflection(session)
            if _DOCUMENTS is None:
                return []
            documents = _DOCUMENTS
            query = select(
                documents.c.id,
                documents.c.filename,
                documents.c.client_name,
                documents.c.period,
                documents.c.status,
                documents.c.executive_summary,
            )
            if client_name:
                query = query.where(documents.c.client_name.ilike(f"%{client_name}%"))
            if status:
                query = query.where(documents.c.status == status)
            query = query.limit(limit)

            result = await session.execute(query)
            rows = result.fetchall()
            return [
                Document(
                    document_id=DocumentId(int(row.id)),
                    filename=row.filename,
                    client_name=row.client_name,
                    period=row.period,
                    status=row.status,
                    executive_summary=row.executive_summary,
                )
                for row in rows
            ]

    async def fetch_chunks_by_ids(self, chunk_ids: Iterable[ChunkId]) -> list[Chunk]:
        ids = [chunk_id.value for chunk_id in chunk_ids]
        if not ids:
            return []
        async with self.sessionmaker() as session:
            await self._ensure_reflection(session)
            if _CHUNKS is None:
                return []
            chunks = _CHUNKS
            query = select(
                chunks.c.id,
                chunks.c.document_id,
                chunks.c.content,
                chunks.c.start_slide,
                chunks.c.end_slide,
                chunks.c.summary,
                chunks.c.topics,
                chunks.c.importance_score,
                chunks.c.embedding,
                chunks.c.embedding_model,
            ).where(chunks.c.id.in_(ids))
            result = await session.execute(query)
            rows = result.fetchall()
            return [self._row_to_chunk(row) for row in rows]

    async def fetch_chunk_embeddings(
        self,
        *,
        document_id: DocumentId | None = None,
        embedding_model: str | None = None,
    ) -> list[tuple[ChunkId, EmbeddingVector]]:
        async with self.sessionmaker() as session:
            await self._ensure_reflection(session)
            if _CHUNKS is None:
                return []
            chunks = _CHUNKS
            query = select(chunks.c.id, chunks.c.embedding)
            query = query.where(chunks.c.embedding.is_not(None))
            if document_id is not None:
                query = query.where(chunks.c.document_id == document_id.value)
            if embedding_model:
                query = query.where(chunks.c.embedding_model == embedding_model)
            result = await session.execute(query)
            rows = result.fetchall()
            pairs: list[tuple[ChunkId, EmbeddingVector]] = []
            for row in rows:
                vector = _coerce_vector(row.embedding)
                if vector:
                    pairs.append((ChunkId(int(row.id)), EmbeddingVector(tuple(vector))))
            return pairs

    async def search_chunks_text(
        self,
        *,
        query: str,
        limit: int,
        document_id: DocumentId | None = None,
    ) -> list[TextMatch]:
        terms = [term for term in query.lower().split() if term]
        if not terms:
            return []
        async with self.sessionmaker() as session:
            await self._ensure_reflection(session)
            if _CHUNKS is None:
                return []
            chunks = _CHUNKS
            conditions = [chunks.c.content.ilike(f"%{term}%") for term in terms]
            stmt = select(chunks.c.id, chunks.c.content).where(or_(*conditions)).limit(limit)
            if document_id is not None:
                stmt = stmt.where(chunks.c.document_id == document_id.value)
            result = await session.execute(stmt)
            rows = result.fetchall()
            matches: list[TextMatch] = []
            for row in rows:
                content = str(row.content).lower()
                score = sum(1.0 for term in terms if term in content)
                matches.append(TextMatch(chunk_id=ChunkId(int(row.id)), score=score))
            return matches

    def _row_to_chunk(self, row: Any) -> Chunk:
        embedding = None
        raw_embedding = _coerce_vector(getattr(row, "embedding", None))
        if raw_embedding:
            embedding = Embedding(
                vector=EmbeddingVector(tuple(raw_embedding)),
                model=getattr(row, "embedding_model", None),
            )
        topics = getattr(row, "topics", None)
        if isinstance(topics, list):
            topics_list = [str(item) for item in topics]
        else:
            topics_list = None
        return Chunk(
            chunk_id=ChunkId(int(row.id)),
            document_id=DocumentId(int(row.document_id)),
            content=str(row.content),
            start_slide=getattr(row, "start_slide", None),
            end_slide=getattr(row, "end_slide", None),
            summary=getattr(row, "summary", None),
            topics=topics_list,
            importance_score=getattr(row, "importance_score", None),
            embedding=embedding,
        )


def _coerce_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    return []

