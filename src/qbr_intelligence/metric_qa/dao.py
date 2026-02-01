"""Metric fact store access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

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
    m.slide_id AS slide_id,
    s.slide_number AS slide_number,
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
    slide_id: int | None
    slide_number: int | None
    confidence: float | None
    label_text: str | None
    raw_value_text: str | None
    snippet: str | None
    llm_context_label: str | None

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
            await conn.execute(text(_VIEW_SQL))
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
        client_name: str | None = None,
        region: str | None = None,
        period_start: date | None = None,
        period_end: date | None = None,
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
            clauses.append("client_name LIKE :client_name")
            params["client_name"] = f"%{client_name}%"
        if region:
            clauses.append("region = :region")
            params["region"] = region
        if period_start and period_end:
            clauses.append("period_start IS NOT NULL AND period_end IS NOT NULL")
            clauses.append("period_end >= :period_start")
            clauses.append("period_start <= :period_end")
            params["period_start"] = period_start.isoformat()
            params["period_end"] = period_end.isoformat()
        where_clause = " AND ".join(clauses)
        sql_text = f"SELECT * FROM metric_fact WHERE {where_clause} ORDER BY {order_by} LIMIT :limit"
        params["limit"] = int(limit)
        stmt = text(sql_text)
        if metric_ids:
            stmt = stmt.bindparams(bindparam("metric_ids", expanding=True))
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt, params)
            rows = [MetricFactRow.from_row(dict(row._mapping)) for row in result]
        return rows, sql_text, params
