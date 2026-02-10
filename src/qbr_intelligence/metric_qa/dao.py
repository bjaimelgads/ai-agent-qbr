"""Metric fact store access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
import json
import re

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine

from qbr_agent.infrastructure.sqlalchemy_gateway import DatabaseGateway


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
        self._engine: AsyncEngine = DatabaseGateway(database_url=database_url).engine()
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
            dialect = conn.engine.dialect.name
            try:
                if dialect == "sqlite":
                    result = await conn.execute(text("PRAGMA table_info('slides');"))
                    columns = {row[1] for row in result.fetchall()}
                else:
                    result = await conn.execute(
                        text(
                            """
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = 'slides'
                            """
                        )
                    )
                    columns = {row[0] for row in result.fetchall()}
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

    async def query_document_ids(
        self,
        *,
        metric_ids: list[str] | None = None,
        client_name: list[str] | None = None,
        region: list[str] | None = None,
        period_ranges: list[tuple[date, date]] | None = None,
        period_specs: list[dict[str, Any]] | None = None,
        limit: int = 100,
    ) -> tuple[list[int], str, dict[str, Any]]:
        await self.ensure_view()
        clauses = ["1=1"]
        params: dict[str, Any] = {}
        if client_name:
            client_clauses = []
            for idx, name in enumerate(client_name):
                key = f"client_name_{idx}"
                client_clauses.append(f"c.name LIKE :{key}")
                params[key] = f"%{name}%"
            if client_clauses:
                clauses.append("(" + " OR ".join(client_clauses) + ")")
        if region:
            clauses.append("r.code IN :regions")
            params["regions"] = region
        if metric_ids:
            clauses.append(
                "("
                "EXISTS ("
                "SELECT 1 FROM metrics m "
                "LEFT JOIN metric_catalog mc ON mc.id = m.metric_catalog_id "
                "WHERE m.document_id = d.id "
                "AND COALESCE(mc.slug, m.name) IN :metric_ids"
                ")"
                ")"
            )
            params["metric_ids"] = metric_ids
        doc_period_clauses = self._build_document_period_clauses(period_specs, params)
        if doc_period_clauses:
            clauses.append("(" + " OR ".join(doc_period_clauses) + ")")
        if period_ranges:
            range_clauses = []
            for idx, (start, end) in enumerate(period_ranges):
                start_key = f"period_start_{idx}"
                end_key = f"period_end_{idx}"
                range_clauses.append(
                    "(EXISTS ("
                    "SELECT 1 FROM metrics m "
                    "LEFT JOIN periods p ON p.id = m.period_id "
                    "WHERE m.document_id = d.id "
                    f"AND COALESCE(m.period_start, p.start_date) IS NOT NULL "
                    f"AND COALESCE(m.period_end, p.end_date) IS NOT NULL "
                    f"AND COALESCE(m.period_end, p.end_date) >= :{start_key} "
                    f"AND COALESCE(m.period_start, p.start_date) <= :{end_key}"
                    "))"
                )
                params[start_key] = start.isoformat()
                params[end_key] = end.isoformat()
            if range_clauses:
                clauses.append("(" + " OR ".join(range_clauses) + ")")
        where_clause = " AND ".join(clauses)
        sql_text = (
            "SELECT DISTINCT d.id AS document_id "
            "FROM documents d "
            "LEFT JOIN clients c ON c.id = d.client_id "
            "LEFT JOIN regions r ON r.id = d.region_id "
            f"WHERE {where_clause} "
            "ORDER BY d.id ASC LIMIT :limit"
        )
        params["limit"] = int(limit)
        stmt = text(sql_text)
        if region:
            stmt = stmt.bindparams(bindparam("regions", expanding=True))
        if metric_ids:
            stmt = stmt.bindparams(bindparam("metric_ids", expanding=True))
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt, params)
            rows = [int(row[0]) for row in result.fetchall() if row[0] is not None]
        return rows, sql_text, params

    def _build_document_period_clauses(
        self,
        period_specs: list[dict[str, Any]] | None,
        params: dict[str, Any],
    ) -> list[str]:
        if not period_specs:
            return []

        clauses: list[str] = []
        for idx, raw in enumerate(period_specs):
            if not isinstance(raw, dict):
                continue
            ptype = str(raw.get("type") or "").lower()
            pvalue = str(raw.get("value") or "").upper()
            start = raw.get("start")
            end = raw.get("end")
            fy_candidates = self._derive_fy_candidates(pvalue, start, end)
            if ptype == "half":
                half = self._extract_half(pvalue)
                if not half:
                    continue
                half_key = f"doc_half_{idx}"
                params[half_key] = half
                fy_checks = []
                for fy_idx, fy in enumerate(fy_candidates):
                    fy_key = f"doc_fy_{idx}_{fy_idx}"
                    params[fy_key] = fy
                    fy_checks.append(f"d.fiscal_year = :{fy_key}")
                    fy_checks.append(f"d.report_period LIKE :{fy_key}_report")
                    params[f"{fy_key}_report"] = f"%{fy}%"
                fy_clause = "(" + " OR ".join(fy_checks) + ")" if fy_checks else "1=1"
                clauses.append(
                    "("
                    f"(d.half = :{half_key} AND {fy_clause}) "
                    f"OR (d.report_period LIKE :{half_key}_report AND {fy_clause})"
                    ")"
                )
                params[f"{half_key}_report"] = f"%{half}%"
                continue

            if ptype == "quarter":
                quarter = self._extract_quarter(pvalue)
                if not quarter:
                    continue
                q_key = f"doc_quarter_{idx}"
                params[q_key] = quarter
                fy_checks = []
                for fy_idx, fy in enumerate(fy_candidates):
                    fy_key = f"doc_fy_{idx}_{fy_idx}"
                    params[fy_key] = fy
                    fy_checks.append(f"d.fiscal_year = :{fy_key}")
                    fy_checks.append(f"d.report_period LIKE :{fy_key}_report")
                    params[f"{fy_key}_report"] = f"%{fy}%"
                fy_clause = "(" + " OR ".join(fy_checks) + ")" if fy_checks else "1=1"
                clauses.append(
                    "("
                    f"(d.quarter = :{q_key} AND {fy_clause}) "
                    f"OR (d.report_period LIKE :{q_key}_report AND {fy_clause})"
                    ")"
                )
                params[f"{q_key}_report"] = f"%{quarter}%"
                continue

            if ptype == "year":
                fy_checks = []
                for fy_idx, fy in enumerate(fy_candidates):
                    fy_key = f"doc_fy_{idx}_{fy_idx}"
                    params[fy_key] = fy
                    fy_checks.append(f"d.fiscal_year = :{fy_key}")
                    fy_checks.append(f"d.report_period LIKE :{fy_key}_report")
                    params[f"{fy_key}_report"] = f"%{fy}%"
                if fy_checks:
                    clauses.append("(" + " OR ".join(fy_checks) + ")")

        return clauses

    @staticmethod
    def _derive_fy_candidates(value: str, start: Any, end: Any) -> list[str]:
        candidates: list[str] = []
        value = (value or "").upper()
        explicit_matches = re.findall(r"FY\s*(\d{2,4})", value)
        for match in explicit_matches:
            if len(match) == 2:
                candidates.append(f"FY{match}")
            else:
                candidates.append(f"FY{match[-2:]}")
        if candidates:
            return list(dict.fromkeys(candidates))
        if isinstance(end, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", end):
            candidates.append(f"FY{end[2:4]}")
        if not candidates and isinstance(start, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", start):
            # Fallback when end is missing.
            candidates.append(f"FY{start[2:4]}")
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _extract_half(value: str) -> str | None:
        if "H1" in value:
            return "H1"
        if "H2" in value:
            return "H2"
        return None

    @staticmethod
    def _extract_quarter(value: str) -> str | None:
        for q in ("Q1", "Q2", "Q3", "Q4"):
            if q in value:
                return q
        return None

    async def query_slide_ids_for_ranges(
        self,
        *,
        ranges: list[tuple[int, int | None, int | None]],
        limit: int = 5000,
    ) -> tuple[list[int], str, dict[str, Any]]:
        await self.ensure_view()
        if not ranges:
            return [], "SELECT id FROM slides WHERE 1=0", {}

        clauses: list[str] = []
        params: dict[str, Any] = {"limit": int(limit)}
        for idx, (document_id, start_slide, end_slide) in enumerate(ranges):
            doc_key = f"doc_{idx}"
            start_key = f"start_{idx}"
            end_key = f"end_{idx}"
            params[doc_key] = int(document_id)
            if start_slide is None and end_slide is None:
                clauses.append(f"(document_id = :{doc_key})")
                continue
            if start_slide is None:
                clauses.append(f"(document_id = :{doc_key} AND slide_number <= :{end_key})")
                params[end_key] = int(end_slide)
                continue
            if end_slide is None:
                clauses.append(f"(document_id = :{doc_key} AND slide_number >= :{start_key})")
                params[start_key] = int(start_slide)
                continue
            clauses.append(
                f"(document_id = :{doc_key} AND slide_number >= :{start_key} AND slide_number <= :{end_key})"
            )
            params[start_key] = int(start_slide)
            params[end_key] = int(end_slide)

        where = " OR ".join(clauses) if clauses else "1=0"
        sql_text = f"SELECT DISTINCT id FROM slides WHERE {where} ORDER BY id ASC LIMIT :limit"
        stmt = text(sql_text)
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt, params)
            slide_ids = [int(row[0]) for row in result.fetchall() if row[0] is not None]
        return slide_ids, sql_text, params

    async def query_document_names(self, *, document_ids: list[int]) -> dict[int, str | None]:
        await self.ensure_view()
        if not document_ids:
            return {}
        stmt = (
            text(
                """
                SELECT document_id, MAX(document_name) AS document_name
                FROM metric_fact
                WHERE document_id IN :document_ids
                GROUP BY document_id
                """
            )
            .bindparams(bindparam("document_ids", expanding=True))
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(stmt, {"document_ids": [int(doc_id) for doc_id in document_ids]})
            return {int(row[0]): row[1] for row in result.fetchall() if row[0] is not None}
