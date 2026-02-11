"""SQLAlchemy repository for QBR knowledge."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace, dataclass, field
import json
import logging
from typing import Any

from sqlalchemy import MetaData, Table, literal, or_, select, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from qbr_agent.application.ports import KnowledgeRepository, TextMatch
from qbr_agent.domain.entities import Chunk, Document, Embedding
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector

_METADATA = MetaData()

_DOCUMENTS: Table | None = None
_CHUNKS: Table | None = None
_SLIDES: Table | None = None
_CLIENTS: Table | None = None
_PERIODS: Table | None = None
_CHUNKS_FTS: Table | None = None
_WARNED_MISSING_TABLES = False
_WARNED_MISSING_FTS = False

_LOGGER = logging.getLogger(__name__)


@dataclass
class SqlAlchemyKnowledgeRepository(KnowledgeRepository):
    sessionmaker: async_sessionmaker[AsyncSession]
    text_search_backend: str = "fts5"
    fts_table: str = "chunks_fts"
    _db_backend: str | None = field(default=None, init=False, repr=False)

    async def _ensure_reflection(self, session: AsyncSession) -> None:
        global _DOCUMENTS, _CHUNKS, _SLIDES, _CLIENTS, _PERIODS, _CHUNKS_FTS, _WARNED_MISSING_TABLES
        if (
            _DOCUMENTS is not None
            and _CHUNKS is not None
            and _SLIDES is not None
            and (
                _CHUNKS_FTS is not None
                or self.text_search_backend in {"like", "auto", "postgres_fts"}
            )
        ):
            return

        conn = await session.connection()
        if self._db_backend is None:
            self._db_backend = conn.engine.dialect.name
        await conn.run_sync(_METADATA.reflect)
        _DOCUMENTS = _METADATA.tables.get("documents")
        _CHUNKS = _METADATA.tables.get("chunks")
        _SLIDES = _METADATA.tables.get("slides")
        _CLIENTS = _METADATA.tables.get("clients")
        _PERIODS = _METADATA.tables.get("periods")
        if self._db_backend == "sqlite":
            _CHUNKS_FTS = _METADATA.tables.get(self.fts_table)
        else:
            _CHUNKS_FTS = None
        if not _WARNED_MISSING_TABLES and (
            _DOCUMENTS is None or _CHUNKS is None or _SLIDES is None
        ):
            missing = [
                name
                for name, table in (
                    ("documents", _DOCUMENTS),
                    ("chunks", _CHUNKS),
                    ("slides", _SLIDES),
                )
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
            clients = _CLIENTS
            periods = _PERIODS
            file_path_col = documents.c.get("file_path")
            if file_path_col is None:
                file_path_col = literal(None).label("file_path")
            period_col = documents.c.get("period")
            if period_col is None:
                if periods is not None and documents.c.get("report_period_id") is not None:
                    period_col = periods.c.get("period_label")
                else:
                    period_col = documents.c.get("report_period")
            if period_col is None:
                period_col = literal(None).label("period")
            client_name_col = None
            if clients is not None:
                client_name_col = clients.c.get("name")
            if client_name_col is None:
                client_name_col = documents.c.get("client_name")
            if client_name_col is None:
                client_name_col = literal(None).label("client_name")
            query = select(
                documents.c.id,
                documents.c.filename,
                file_path_col,
                client_name_col.label("client_name"),
                period_col,
                documents.c.status,
                documents.c.executive_summary,
            )
            if clients is not None and documents.c.get("client_id") is not None:
                query = query.outerjoin(
                    clients,
                    clients.c.id == documents.c.client_id,
                )
            if periods is not None and documents.c.get("report_period_id") is not None:
                query = query.outerjoin(
                    periods,
                    periods.c.id == documents.c.report_period_id,
                )
            if client_name:
                query = query.where(client_name_col.ilike(f"%{client_name}%"))
            if status:
                query = query.where(documents.c.status == status)
            query = query.limit(limit)

            result = await session.execute(query)
            rows = result.fetchall()
            return [
                Document(
                    document_id=DocumentId(int(row.id)),
                    filename=row.filename,
                    file_path=getattr(row, "file_path", None),
                    client_name=row.client_name,
                    period=getattr(row, "period", None) or getattr(row, "report_period", None),
                    status=row.status,
                    executive_summary=row.executive_summary,
                )
                for row in rows
            ]

    async def fetch_documents_by_ids(
        self,
        document_ids: Iterable[DocumentId],
    ) -> list[Document]:
        ids = [document_id.value for document_id in document_ids]
        if not ids:
            return []
        async with self.sessionmaker() as session:
            await self._ensure_reflection(session)
            if _DOCUMENTS is None:
                return []
            documents = _DOCUMENTS
            clients = _CLIENTS
            periods = _PERIODS
            file_path_col = documents.c.get("file_path")
            if file_path_col is None:
                file_path_col = literal(None).label("file_path")
            period_col = documents.c.get("period")
            if period_col is None:
                if periods is not None and documents.c.get("report_period_id") is not None:
                    period_col = periods.c.get("period_label")
                else:
                    period_col = documents.c.get("report_period")
            if period_col is None:
                period_col = literal(None).label("period")
            client_name_col = None
            if clients is not None:
                client_name_col = clients.c.get("name")
            if client_name_col is None:
                client_name_col = documents.c.get("client_name")
            if client_name_col is None:
                client_name_col = literal(None).label("client_name")
            query = select(
                documents.c.id,
                documents.c.filename,
                file_path_col,
                client_name_col.label("client_name"),
                period_col,
                documents.c.status,
                documents.c.executive_summary,
            ).where(documents.c.id.in_(ids))
            if clients is not None and documents.c.get("client_id") is not None:
                query = query.outerjoin(
                    clients,
                    clients.c.id == documents.c.client_id,
                )
            if periods is not None and documents.c.get("report_period_id") is not None:
                query = query.outerjoin(
                    periods,
                    periods.c.id == documents.c.report_period_id,
                )
            result = await session.execute(query)
            rows = result.fetchall()
            return [
                Document(
                    document_id=DocumentId(int(row.id)),
                    filename=row.filename,
                    file_path=getattr(row, "file_path", None),
                    client_name=row.client_name,
                    period=getattr(row, "period", None) or getattr(row, "report_period", None),
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
            slide_metadata = await self._fetch_slide_metadata(session, rows)
            chunks_out: list[Chunk] = []
            for row in rows:
                chunk = self._row_to_chunk(row)
                meta = slide_metadata.get(
                    (int(row.document_id), getattr(row, "start_slide", None))
                )
                if meta:
                    chunk = replace(chunk, metadata=meta)
                chunks_out.append(chunk)
            return chunks_out

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
            backend = self.text_search_backend.lower()
            backend = self._select_text_backend(backend)
            if backend == "fts5":
                matches = await self._search_chunks_fts5(
                    session=session,
                    terms=terms,
                    query=query,
                    limit=limit,
                    document_id=document_id,
                )
                if matches is not None:
                    if _LOGGER.isEnabledFor(logging.DEBUG):
                        _LOGGER.debug(
                            "FTS5 text search returned %d matches (limit=%d).",
                            len(matches),
                            limit,
                        )
                    return matches
            if backend == "postgres_fts":
                matches = await self._search_chunks_postgres_fts(
                    session=session,
                    terms=terms,
                    query=query,
                    limit=limit,
                    document_id=document_id,
                )
                if matches is not None:
                    return matches
            return await self._search_chunks_like(
                session=session,
                terms=terms,
                limit=limit,
                document_id=document_id,
            )

    def _select_text_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        if self._db_backend is None:
            return "like"
        if self._db_backend == "sqlite":
            return "fts5"
        if self._db_backend in {"postgresql", "postgres"}:
            return "postgres_fts"
        return "like"

    async def _search_chunks_fts5(
        self,
        *,
        session: AsyncSession,
        terms: list[str],
        query: str,
        limit: int,
        document_id: DocumentId | None,
    ) -> list[TextMatch] | None:
        global _WARNED_MISSING_FTS
        if _CHUNKS is None or _CHUNKS_FTS is None:
            if not _WARNED_MISSING_FTS:
                _LOGGER.warning(
                    "FTS table %s missing; falling back to LIKE search.",
                    self.fts_table,
                )
                _WARNED_MISSING_FTS = True
            return None
        fts_query = " ".join(terms)
        doc_clause = ""
        params: dict[str, Any] = {"query": fts_query, "limit": limit}
        if document_id is not None:
            doc_clause = " AND chunks.document_id = :document_id"
            params["document_id"] = document_id.value
        stmt = text(
            f"""
            SELECT chunks.id AS id, bm25({_CHUNKS_FTS.name}) AS bm25_score
            FROM {_CHUNKS_FTS.name}
            JOIN chunks ON {_CHUNKS_FTS.name}.rowid = chunks.id
            WHERE {_CHUNKS_FTS.name} MATCH :query{doc_clause}
            ORDER BY bm25_score ASC
            LIMIT :limit
            """
        )
        try:
            result = await session.execute(stmt, params)
        except OperationalError as exc:
            if not _WARNED_MISSING_FTS:
                _LOGGER.warning("FTS query failed; falling back to LIKE. %s", exc)
                _WARNED_MISSING_FTS = True
            return None
        rows = result.fetchall()
        matches: list[TextMatch] = []
        for row in rows:
            bm25_score = float(row.bm25_score) if row.bm25_score is not None else 0.0
            score = 1.0 / (1.0 + bm25_score)
            matches.append(TextMatch(chunk_id=ChunkId(int(row.id)), score=score))
        if _LOGGER.isEnabledFor(logging.DEBUG):
            top_score = matches[0].score if matches else None
            _LOGGER.debug("FTS5 BM25 top score: %s", top_score)
        return matches

    async def _search_chunks_postgres_fts(
        self,
        *,
        session: AsyncSession,
        terms: list[str],
        query: str,
        limit: int,
        document_id: DocumentId | None,
    ) -> list[TextMatch] | None:
        if _CHUNKS is None:
            return None
        params: dict[str, Any] = {"query": query, "limit": limit}
        doc_clause = ""
        if document_id is not None:
            doc_clause = " AND chunks.document_id = :document_id"
            params["document_id"] = document_id.value
        stmt = text(
            f"""
            SELECT chunks.id AS id,
                   ts_rank_cd(
                       to_tsvector('english', chunks.content),
                       plainto_tsquery('english', :query)
                   ) AS rank
            FROM chunks
            WHERE to_tsvector('english', chunks.content)
                  @@ plainto_tsquery('english', :query){doc_clause}
            ORDER BY rank DESC
            LIMIT :limit
            """
        )
        try:
            result = await session.execute(stmt, params)
        except OperationalError as exc:
            _LOGGER.warning("Postgres FTS query failed; falling back to LIKE. %s", exc)
            return None
        rows = result.fetchall()
        matches: list[TextMatch] = []
        for row in rows:
            rank = float(row.rank) if row.rank is not None else 0.0
            matches.append(TextMatch(chunk_id=ChunkId(int(row.id)), score=rank))
        if _LOGGER.isEnabledFor(logging.DEBUG):
            top_score = matches[0].score if matches else None
            _LOGGER.debug("Postgres FTS top score: %s", top_score)
        return matches

    async def _search_chunks_like(
        self,
        *,
        session: AsyncSession,
        terms: list[str],
        limit: int,
        document_id: DocumentId | None,
    ) -> list[TextMatch]:
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
        if _LOGGER.isEnabledFor(logging.DEBUG):
            top_score = matches[0].score if matches else None
            _LOGGER.debug("LIKE text search returned %d matches, top score=%s.", len(matches), top_score)
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

    async def _fetch_slide_metadata(
        self,
        session: AsyncSession,
        rows: list[Any],
    ) -> dict[tuple[int, int | None], dict[str, Any]]:
        if _SLIDES is None:
            return {}
        slide_keys: dict[int, set[int]] = {}
        for row in rows:
            slide_num = getattr(row, "start_slide", None)
            if slide_num is None:
                continue
            slide_keys.setdefault(int(row.document_id), set()).add(int(slide_num))
        if not slide_keys:
            return {}

        slides = _SLIDES
        metadata: dict[tuple[int, int], dict[str, Any]] = {}
        for document_id, slide_numbers in slide_keys.items():
            query = select(
                slides.c.document_id,
                slides.c.slide_number,
                slides.c.title,
                slides.c.key_message,
                slides.c.slide_type,
                slides.c.insights,
                slides.c.action_items,
            ).where(
                slides.c.document_id == document_id,
                slides.c.slide_number.in_(slide_numbers),
            )
            result = await session.execute(query)
            for row in result.fetchall():
                metadata[(int(row.document_id), int(row.slide_number))] = {
                    "slide_number": int(row.slide_number),
                    "slide_title": row.title,
                    "slide_key_message": row.key_message,
                    "slide_type": row.slide_type,
                    "slide_insights": row.insights,
                    "slide_action_items": row.action_items,
                }
        return metadata


def _coerce_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        try:
            decoded = value.decode("utf-8")
        except Exception:
            return []
        try:
            parsed = json.loads(decoded)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    return []
