"""Metric fact store access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
import json

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


_VIEW_SQL = """
CREATE VIEW metric_fact AS
SELECT
    m.id AS fact_id,
    COALESCE(mc.slug, m.name) AS metric_id,
    COALESCE(mc.name, m.name) AS metric_name,
    m.normalized_value AS value,
    m.unit AS unit,
    'ones' AS scale,
    d.client_id AS client_id,
    c.name AS client_name,
    r.code AS region,
    COALESCE(m.period_start, p.start_date) AS period_start,
    COALESCE(m.period_end, p.end_date) AS period_end,
    p.period_type AS period_granularity,
    COALESCE(m.period_label, p.period_label) AS period_label,
    m.document_id AS document_id,
    d.filename AS document_name,
    d.file_path AS document_url,
    m.slide_id AS slide_id,
    s.slide_number AS slide_number,
    s.title AS slide_title,
    s.google_slide_id AS google_slide_id,
    m.extraction_confidence AS confidence,
    COALESCE(m.name, mc.name) AS label_text,
    m.raw_value AS raw_value_text,
    m.raw_context AS snippet,
    m.llm_context_label AS llm_context_label
FROM metrics m
LEFT JOIN metric_catalog mc ON mc.id = m.metric_catalog_id
LEFT JOIN documents d ON d.id = m.document_id
LEFT JOIN clients c ON c.id = d.client_id
LEFT JOIN regions r ON r.id = m.region_id
LEFT JOIN periods p ON p.id = m.period_id
LEFT JOIN slides s ON s.id = m.slide_id;
"""

_VIEW_SQL_NO_GOOGLE = """
CREATE VIEW metric_fact AS
SELECT
    m.id AS fact_id,
    COALESCE(mc.slug, m.name) AS metric_id,
    COALESCE(mc.name, m.name) AS metric_name,
    m.normalized_value AS value,
    m.unit AS unit,
    'ones' AS scale,
    d.client_id AS client_id,
    c.name AS client_name,
    r.code AS region,
    COALESCE(m.period_start, p.start_date) AS period_start,
    COALESCE(m.period_end, p.end_date) AS period_end,
    p.period_type AS period_granularity,
    COALESCE(m.period_label, p.period_label) AS period_label,
    m.document_id AS document_id,
    d.filename AS document_name,
    d.file_path AS document_url,
    m.slide_id AS slide_id,
    s.slide_number AS slide_number,
    s.title AS slide_title,
    NULL AS google_slide_id,
    m.extraction_confidence AS confidence,
    COALESCE(m.name, mc.name) AS label_text,
    m.raw_value AS raw_value_text,
    m.raw_context AS snippet,
    m.llm_context_label AS llm_context_label
FROM metrics m
LEFT JOIN metric_catalog mc ON mc.id = m.metric_catalog_id
LEFT JOIN documents d ON d.id = m.document_id
LEFT JOIN clients c ON c.id = d.client_id
LEFT JOIN regions r ON r.id = m.region_id
LEFT JOIN periods p ON p.id = m.period_id
LEFT JOIN slides s ON s.id = m.slide_id;
"""

_VIEW_SQL_NO_TITLE = """
CREATE VIEW metric_fact AS
SELECT
    m.id AS fact_id,
    COALESCE(mc.slug, m.name) AS metric_id,
    COALESCE(mc.name, m.name) AS metric_name,
    m.normalized_value AS value,
    m.unit AS unit,
    'ones' AS scale,
    d.client_id AS client_id,
    c.name AS client_name,
    r.code AS region,
    COALESCE(m.period_start, p.start_date) AS period_start,
    COALESCE(m.period_end, p.end_date) AS period_end,
    p.period_type AS period_granularity,
    COALESCE(m.period_label, p.period_label) AS period_label,
    m.document_id AS document_id,
    d.filename AS document_name,
    d.file_path AS document_url,
    m.slide_id AS slide_id,
    s.slide_number AS slide_number,
    NULL AS slide_title,
    s.google_slide_id AS google_slide_id,
    m.extraction_confidence AS confidence,
    COALESCE(m.name, mc.name) AS label_text,
    m.raw_value AS raw_value_text,
    m.raw_context AS snippet,
    m.llm_context_label AS llm_context_label
FROM metrics m
LEFT JOIN metric_catalog mc ON mc.id = m.metric_catalog_id
LEFT JOIN documents d ON d.id = m.document_id
LEFT JOIN clients c ON c.id = d.client_id
LEFT JOIN regions r ON r.id = m.region_id
LEFT JOIN periods p ON p.id = m.period_id
LEFT JOIN slides s ON s.id = m.slide_id;
"""

_VIEW_SQL_NO_TITLE_NO_GOOGLE = """
CREATE VIEW metric_fact AS
SELECT
    m.id AS fact_id,
    COALESCE(mc.slug, m.name) AS metric_id,
    COALESCE(mc.name, m.name) AS metric_name,
    m.normalized_value AS value,
    m.unit AS unit,
    'ones' AS scale,
    d.client_id AS client_id,
    c.name AS client_name,
    r.code AS region,
    COALESCE(m.period_start, p.start_date) AS period_start,
    COALESCE(m.period_end, p.end_date) AS period_end,
    p.period_type AS period_granularity,
    COALESCE(m.period_label, p.period_label) AS period_label,
    m.document_id AS document_id,
    d.filename AS document_name,
    d.file_path AS document_url,
    m.slide_id AS slide_id,
    s.slide_number AS slide_number,
    NULL AS slide_title,
    NULL AS google_slide_id,
    m.extraction_confidence AS confidence,
    COALESCE(m.name, mc.name) AS label_text,
    m.raw_value AS raw_value_text,
    m.raw_context AS snippet,
    m.llm_context_label AS llm_context_label
FROM metrics m
LEFT JOIN metric_catalog mc ON mc.id = m.metric_catalog_id
LEFT JOIN documents d ON d.id = m.document_id
LEFT JOIN clients c ON c.id = d.client_id
LEFT JOIN regions r ON r.id = m.region_id
LEFT JOIN periods p ON p.id = m.period_id
LEFT JOIN slides s ON s.id = m.slide_id;
"""


@dataclass
class MetricFactRow:
    fact_id: int
    metric_id: str | None
    metric_name: str | None
    value: float | None
    unit: str | None
    scale: str | None
    client_id: int | None
    client_name: str | None
    region: str | None
    period_start: str | None
    period_end: str | None
    period_granularity: str | None
    period_label: str | None
    document_id: int
    document_name: str | None
    document_url: str | None
    slide_id: int | None
    slide_number: int | None
    slide_title: str | None
    google_slide_id: str | None
    confidence: float | None
    label_text: str | None
    raw_value_text: str | None
    snippet: str | None
    llm_context_label: str | None
    semantic_score: float | None = None

    @classmethod
    def from_row(cls, row: Any) -> "MetricFactRow":
        return cls(**row)


class MetricFactStore:
    def __init__(self, database_url: str) -> None:
        self._engine: AsyncEngine = create_async_engine(database_url)
        self._view_ready = False

    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    async def ensure_view(self) -> None:
        if self._view_ready:
            return
        async with self._engine.begin() as conn:
            await conn.execute(text("DROP VIEW IF EXISTS metric_fact;"))
            has_google_slide_id = False
            has_slide_title = False
            try:
                result = await conn.execute(text("PRAGMA table_info('slides');"))
                columns = {row[1] for row in result.fetchall()}
                has_google_slide_id = "google_slide_id" in columns
                has_slide_title = "title" in columns
            except Exception:
                has_google_slide_id = False
                has_slide_title = False
            if has_google_slide_id and has_slide_title:
                sql = _VIEW_SQL
            elif has_google_slide_id and not has_slide_title:
                sql = _VIEW_SQL_NO_TITLE
            elif not has_google_slide_id and has_slide_title:
                sql = _VIEW_SQL_NO_GOOGLE
            else:
                sql = _VIEW_SQL_NO_TITLE_NO_GOOGLE
            await conn.execute(text(sql))
        self._view_ready = True

    async def fetch_metric_catalog(self) -> list[dict[str, Any]]:
        await self.ensure_view()
        sql = text(
            """
            SELECT mc.id, mc.name, mc.slug, mc.category, mc.default_unit,
                   ma.alias, ma.pattern, ma.priority, ma.unit_override
            FROM metric_catalog mc
            LEFT JOIN metric_aliases ma ON ma.metric_id = mc.id
            ORDER BY mc.id
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql)
            return [dict(row._mapping) for row in result]

    async def fetch_clients(self) -> list[str]:
        await self.ensure_view()
        sql = text("SELECT name FROM clients ORDER BY name")
        async with self._engine.connect() as conn:
            result = await conn.execute(sql)
            return [row[0] for row in result if row[0]]

    async def fetch_metric_definitions(self, metric_ids: list[str]) -> list[dict[str, Any]]:
        await self.ensure_view()
        sql = text(
            """
            SELECT slug, name, description, formula, default_unit
            FROM metric_catalog
            WHERE slug IN :metric_ids OR name IN :metric_ids
            """
        ).bindparams(bindparam("metric_ids", expanding=True))
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"metric_ids": metric_ids})
            return [dict(row._mapping) for row in result]

    async def fetch_metric_fact_embeddings(
        self,
        fact_ids: list[int],
        *,
        embedding_model: str | None = None,
    ) -> dict[int, list[float]]:
        if not fact_ids:
            return {}
        await self.ensure_view()
        sql = text(
            """
            SELECT metric_id, embedding
            FROM metric_fact_embeddings
            WHERE metric_id IN :fact_ids
            """
        ).bindparams(bindparam("fact_ids", expanding=True))
        params: dict[str, Any] = {"fact_ids": fact_ids}
        if embedding_model:
            sql = text(
                """
                SELECT metric_id, embedding
                FROM metric_fact_embeddings
                WHERE metric_id IN :fact_ids AND embedding_model = :embedding_model
                """
            ).bindparams(bindparam("fact_ids", expanding=True))
            params["embedding_model"] = embedding_model
        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(sql, params)
                rows = [dict(row._mapping) for row in result]
        except Exception:
            return {}
        embeddings: dict[int, list[float]] = {}
        for row in rows:
            metric_id = row.get("metric_id")
            embedding = row.get("embedding")
            if metric_id is None or embedding is None:
                continue
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except Exception:
                    continue
            embeddings[int(metric_id)] = [float(value) for value in embedding]
        return embeddings

    async def fetch_metric_fact_embedding_inputs(
        self,
        *,
        limit: int,
        offset: int,
    ) -> list[dict[str, Any]]:
        await self.ensure_view()
        sql = text(
            """
            SELECT
                fact_id,
                metric_name,
                label_text,
                llm_context_label,
                snippet,
                slide_title,
                period_label,
                client_name,
                region,
                document_name
            FROM metric_fact
            ORDER BY fact_id
            LIMIT :limit OFFSET :offset
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"limit": int(limit), "offset": int(offset)})
            return [dict(row._mapping) for row in result]

    async def fetch_latest_period_end(self) -> date | None:
        await self.ensure_view()
        sql = text(
            "SELECT MAX(period_end) AS latest_end FROM metric_fact WHERE period_end IS NOT NULL"
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql)
            row = result.fetchone()
            if not row:
                return None
            value = row[0]
        if not value:
            return None
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None

    async def fetch_latest_period_end_by_granularity(self, granularity: str) -> date | None:
        await self.ensure_view()
        sql = text(
            """
            SELECT MAX(period_end) AS latest_end
            FROM metric_fact
            WHERE period_end IS NOT NULL AND period_granularity = :granularity
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"granularity": granularity})
            row = result.fetchone()
            if not row:
                return None
            value = row[0]
        if not value:
            return None
        try:
            return date.fromisoformat(str(value))
        except ValueError:
            return None

    async def query_facts(
        self,
        *,
        metric_ids: list[str] | None = None,
        client_name: list[str] | None = None,
        region: list[str] | None = None,
        period_ranges: list[tuple[date, date]] | None = None,
        limit: int = 200,
        order_by: str = "period_end DESC",
    ) -> tuple[list[MetricFactRow], str, dict[str, Any]]:
        await self.ensure_view()
        clauses = ["1=1"]
        params: dict[str, Any] = {}
        if metric_ids:
            clauses.append("metric_id IN :metric_ids")
            params["metric_ids"] = metric_ids
        if client_name:
            client_clauses = []
            for idx, name in enumerate(client_name):
                key = f"client_name_{idx}"
                client_clauses.append(f"client_name LIKE :{key}")
                params[key] = f"%{name}%"
            if client_clauses:
                clauses.append("(" + " OR ".join(client_clauses) + ")")
        if region:
            clauses.append("region IN :regions")
            params["regions"] = region
        if period_ranges:
            range_clauses = []
            for idx, (start, end) in enumerate(period_ranges):
                start_key = f"period_start_{idx}"
                end_key = f"period_end_{idx}"
                range_clauses.append(
                    f"(period_start IS NOT NULL AND period_end IS NOT NULL AND period_end >= :{start_key} AND period_start <= :{end_key})"
                )
                params[start_key] = start.isoformat()
                params[end_key] = end.isoformat()
            if range_clauses:
                clauses.append("(" + " OR ".join(range_clauses) + ")")
        where_clause = " AND ".join(clauses)
        sql_text = f"SELECT * FROM metric_fact WHERE {where_clause} ORDER BY {order_by} LIMIT :limit"
        params["limit"] = int(limit)
        stmt = text(sql_text)
        if metric_ids:
            stmt = stmt.bindparams(bindparam("metric_ids", expanding=True))
        if region:
            stmt = stmt.bindparams(bindparam("regions", expanding=True))
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt, params)
            rows = [MetricFactRow.from_row(dict(row._mapping)) for row in result]
        return rows, sql_text, params
