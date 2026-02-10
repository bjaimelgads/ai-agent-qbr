"""pgvector-backed vector index for QBR retrieval."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from qbr_agent.application.ports import VectorIndex, VectorMatch
from qbr_agent.domain.value_objects import ChunkId, DocumentId, EmbeddingVector


@dataclass
class PgVectorIndex(VectorIndex):
    sessionmaker: async_sessionmaker[AsyncSession]
    _extension_checked: bool = field(default=False, init=False, repr=False)
    _extension_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def search(
        self,
        *,
        vector: EmbeddingVector,
        top_k: int,
        document_id: DocumentId | None = None,
        embedding_model: str | None = None,
        min_score: float | None = None,
    ) -> list[VectorMatch]:
        if top_k <= 0:
            return []

        query_literal = _format_vector(vector.values)
        params: dict[str, object] = {"query": query_literal, "limit": int(top_k)}
        where_clauses = ["chunks.embedding IS NOT NULL"]
        if document_id is not None:
            where_clauses.append("chunks.document_id = :document_id")
            params["document_id"] = document_id.value
        if embedding_model:
            where_clauses.append("chunks.embedding_model = :embedding_model")
            params["embedding_model"] = embedding_model
        where_sql = " AND ".join(where_clauses)

        stmt = text(
            f"""
            SELECT
                chunks.id AS id,
                (((chunks.embedding)::text)::vector <=> (:query)::vector) AS distance
            FROM chunks
            WHERE {where_sql}
            ORDER BY distance ASC
            LIMIT :limit
            """
        )

        async with self.sessionmaker() as session:
            backend = session.bind.dialect.name if session.bind is not None else ""
            if backend not in {"postgresql", "postgres"}:
                raise RuntimeError(
                    f"pgvector backend requires PostgreSQL; got database dialect {backend!r}."
                )
            await self._ensure_extension(session)
            result = await session.execute(stmt, params)
            rows = result.fetchall()

        matches: list[VectorMatch] = []
        for row in rows:
            distance = float(row.distance) if row.distance is not None else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))
            if min_score is not None and score < min_score:
                continue
            matches.append(VectorMatch(chunk_id=ChunkId(int(row.id)), score=score))
        return matches

    async def _ensure_extension(self, session: AsyncSession) -> None:
        if self._extension_checked:
            return
        async with self._extension_lock:
            if self._extension_checked:
                return
            result = await session.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            )
            if result.scalar_one_or_none() is None:
                raise RuntimeError(
                    "pgvector extension is not installed; run `CREATE EXTENSION IF NOT EXISTS vector;`"
                )
            self._extension_checked = True


def _format_vector(vector: tuple[float, ...]) -> str:
    values: list[float] = []
    for item in vector:
        number = float(item)
        if math.isnan(number) or math.isinf(number):
            raise ValueError("Vector values must be finite numbers")
        values.append(number)
    if not values:
        raise ValueError("Vector must contain at least one value")
    payload = ",".join(_format_float(value) for value in values)
    return f"[{payload}]"


def _format_float(value: float) -> str:
    rendered = format(value, ".12f").rstrip("0").rstrip(".")
    return rendered if rendered else "0"
