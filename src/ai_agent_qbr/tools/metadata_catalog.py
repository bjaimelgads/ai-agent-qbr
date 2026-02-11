"""Metadata catalog discovery tool for database access and inventory introspection."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext
from sqlalchemy import text

from ai_agent_qbr.models import MetadataCatalogArgs, MetadataCatalogResult
from ai_agent_qbr.tools.question_normalization import normalize_question_arg
from ai_agent_qbr.tools.status import ToolStatusEmitter
from qbr_intelligence.metric_qa import MetricQueryEngine

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SUGGESTED_METADATA_QUESTIONS = [
    "What tables and views can you access in this database?",
    "What metadata scope do you have right now?",
    "Which documents do we have and who are their clients?",
    "List available QBR decks with region and reporting period.",
    "How many documents do we have per client?",
    "How many documents do we have per region?",
    "Which clients are available in the database?",
    "Which regions and countries are represented?",
    "What periods are available (Q, H, FY)?",
    "What is the latest reporting period available?",
    "Which catalog metrics do we have?",
    "List metric catalog entries with category and default unit.",
    "Show aliases for each catalog metric.",
    "How many metric facts do we have per catalog metric?",
    "How many distinct documents and slides contain each metric?",
    "Show metric coverage by client and region.",
    "What columns exist in the documents table?",
    "What columns exist in the metric_catalog table?",
    "Summarize key row counts across documents, metrics, slides, and chunks.",
    "Which metadata fields can be used to filter retrieval?",
]


@dataclass(frozen=True)
class _SchemaSnapshot:
    dialect: str
    tables: set[str]
    views: set[str]
    columns_by_table: dict[str, set[str]]

    def has_table(self, name: str) -> bool:
        return name in self.tables

    def has_view(self, name: str) -> bool:
        return name in self.views

    def has_column(self, table: str, column: str) -> bool:
        return column in self.columns_by_table.get(table, set())


async def _fetch_rows(conn, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    result = await conn.execute(text(sql), params or {})
    return [dict(row._mapping) for row in result]


async def _inspect_schema(conn) -> _SchemaSnapshot:
    dialect = conn.engine.dialect.name
    tables: set[str] = set()
    views: set[str] = set()
    columns_by_table: dict[str, set[str]] = {}

    if dialect == "sqlite":
        rows = await _fetch_rows(
            conn,
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """,
        )
        for row in rows:
            name = str(row.get("name") or "").strip()
            typ = str(row.get("type") or "").strip().lower()
            if not name:
                continue
            if typ == "view":
                views.add(name)
            else:
                tables.add(name)

        for obj_name in sorted(tables | views):
            if not _IDENTIFIER_RE.match(obj_name):
                continue
            pragma_rows = await _fetch_rows(conn, f"PRAGMA table_info('{obj_name}')")
            columns_by_table[obj_name] = {
                str(row.get("name") or "").strip()
                for row in pragma_rows
                if row.get("name")
            }
    else:
        table_rows = await _fetch_rows(
            conn,
            """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """,
        )
        for row in table_rows:
            name = str(row.get("table_name") or "").strip()
            typ = str(row.get("table_type") or "").strip().upper()
            if not name:
                continue
            if typ == "VIEW":
                views.add(name)
            else:
                tables.add(name)

        column_rows = await _fetch_rows(
            conn,
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
            """,
        )
        for row in column_rows:
            table_name = str(row.get("table_name") or "").strip()
            column_name = str(row.get("column_name") or "").strip()
            if not table_name or not column_name:
                continue
            columns_by_table.setdefault(table_name, set()).add(column_name)

    return _SchemaSnapshot(
        dialect=dialect,
        tables=tables,
        views=views,
        columns_by_table=columns_by_table,
    )


def _extract_topics(query: str) -> set[str]:
    lowered = query.lower()
    topics: set[str] = set()

    if any(token in lowered for token in ("access", "scope", "what data", "what can you access", "available data")):
        topics.add("access")
    if any(token in lowered for token in ("schema", "table", "tables", "view", "column", "database structure")):
        topics.add("schema")
    if any(token in lowered for token in ("document", "documents", "deck", "decks", "ppt", "presentation", "file")):
        topics.add("documents")
    if any(token in lowered for token in ("client", "clients", "advertiser", "advertisers")):
        topics.add("clients")
    if any(token in lowered for token in ("region", "regions", "market", "markets", "country", "countries")):
        topics.add("regions")
    if any(token in lowered for token in ("period", "quarter", "quarters", "fiscal", "fy", "h1", "h2")):
        topics.add("periods")
    if any(token in lowered for token in ("metric", "metrics", "kpi", "catalog", "aliases", "definition", "formula", "unit")):
        topics.add("metrics")
    if any(token in lowered for token in ("coverage", "count", "how many", "inventory", "distribution")):
        topics.add("coverage")

    if not topics:
        topics.add("overview")
    return topics


async def _build_schema_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    tables = sorted(schema.tables)
    views = sorted(schema.views)
    sample_tables = tables[:limit]
    columns: dict[str, list[str]] = {}
    for table_name in sample_tables:
        cols = sorted(schema.columns_by_table.get(table_name, set()))
        columns[table_name] = cols
    return {
        "dialect": schema.dialect,
        "table_count": len(tables),
        "view_count": len(views),
        "tables": sample_tables,
        "views": views[:limit],
        "columns": columns,
    }


async def _build_access_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    key_objects = [
        "documents",
        "clients",
        "regions",
        "periods",
        "metric_catalog",
        "metric_aliases",
        "metrics",
        "slides",
        "chunks",
        "metric_fact",
    ]
    available = [
        name
        for name in key_objects
        if schema.has_table(name) or schema.has_view(name)
    ]

    counts: dict[str, int | None] = {}
    for name in available:
        if not _IDENTIFIER_RE.match(name):
            continue
        try:
            rows = await _fetch_rows(conn, f"SELECT COUNT(*) AS row_count FROM {name}")
            counts[name] = int(rows[0]["row_count"]) if rows else 0
        except Exception:
            counts[name] = None

    return {
        "available_key_objects": available,
        "row_counts": counts,
        "all_tables": sorted(schema.tables)[:limit],
        "all_views": sorted(schema.views)[:limit],
    }


async def _build_documents_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    if not schema.has_table("documents"):
        return {"available": False, "reason": "documents table not found"}

    has_clients = schema.has_table("clients") and schema.has_column("documents", "client_id")
    has_regions = schema.has_table("regions") and schema.has_column("documents", "region_id")
    has_period_fk = schema.has_column("documents", "report_period_id") and schema.has_table("periods")
    has_period_legacy = schema.has_column("documents", "report_period")
    has_period = has_period_fk or has_period_legacy

    select_fields = [
        "d.id AS document_id",
        "d.filename AS filename",
        "d.file_path AS file_path",
    ]
    if has_period:
        if has_period_fk and has_period_legacy:
            select_fields.append("COALESCE(p.period_label, d.report_period) AS report_period")
        elif has_period_fk:
            select_fields.append("p.period_label AS report_period")
        else:
            select_fields.append("d.report_period AS report_period")
    if has_clients:
        select_fields.append("COALESCE(c.name, 'Unknown') AS client")
    if has_regions:
        select_fields.append("COALESCE(r.code, 'Unknown') AS region")

    joins = []
    if has_clients:
        joins.append("LEFT JOIN clients c ON c.id = d.client_id")
    if has_regions:
        joins.append("LEFT JOIN regions r ON r.id = d.region_id")
    if has_period_fk:
        joins.append("LEFT JOIN periods p ON p.id = d.report_period_id")

    docs = await _fetch_rows(
        conn,
        f"""
        SELECT {', '.join(select_fields)}
        FROM documents d
        {' '.join(joins)}
        ORDER BY d.id DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )

    by_client: list[dict[str, Any]] = []
    if has_clients:
        by_client = await _fetch_rows(
            conn,
            """
            SELECT COALESCE(c.name, 'Unknown') AS client, COUNT(*) AS document_count
            FROM documents d
            LEFT JOIN clients c ON c.id = d.client_id
            GROUP BY COALESCE(c.name, 'Unknown')
            ORDER BY document_count DESC, client ASC
            LIMIT :limit
            """,
            {"limit": limit},
        )

    by_region: list[dict[str, Any]] = []
    if has_regions:
        by_region = await _fetch_rows(
            conn,
            """
            SELECT COALESCE(r.code, 'Unknown') AS region, COUNT(*) AS document_count
            FROM documents d
            LEFT JOIN regions r ON r.id = d.region_id
            GROUP BY COALESCE(r.code, 'Unknown')
            ORDER BY document_count DESC, region ASC
            LIMIT :limit
            """,
            {"limit": limit},
        )

    period_values: list[str] = []
    if has_period:
        if has_period_fk:
            period_select = (
                "COALESCE(p.period_label, d.report_period)"
                if has_period_legacy
                else "p.period_label"
            )
            rows = await _fetch_rows(
                conn,
                f"""
                SELECT DISTINCT {period_select} AS report_period
                FROM documents d
                LEFT JOIN periods p ON p.id = d.report_period_id
                WHERE {period_select} IS NOT NULL AND TRIM({period_select}) <> ''
                ORDER BY report_period DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )
        else:
            rows = await _fetch_rows(
                conn,
                """
                SELECT DISTINCT report_period
                FROM documents
                WHERE report_period IS NOT NULL AND TRIM(report_period) <> ''
                ORDER BY report_period DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )
        period_values = [str(row["report_period"]) for row in rows]

    return {
        "available": True,
        "documents": docs,
        "documents_by_client": by_client,
        "documents_by_region": by_region,
        "period_values": period_values,
    }


async def _build_clients_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    if not schema.has_table("clients"):
        return {"available": False, "reason": "clients table not found"}

    clients = await _fetch_rows(
        conn,
        """
        SELECT id AS client_id, name AS client
        FROM clients
        ORDER BY name
        LIMIT :limit
        """,
        {"limit": limit},
    )
    return {"available": True, "clients": clients, "client_count": len(clients)}


async def _build_regions_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    if not schema.has_table("regions"):
        return {"available": False, "reason": "regions table not found"}

    regions = await _fetch_rows(
        conn,
        """
        SELECT id AS region_id, code AS region_code, name AS region_name
        FROM regions
        ORDER BY code
        LIMIT :limit
        """,
        {"limit": limit},
    )

    countries: list[dict[str, Any]] = []
    if schema.has_table("region_countries"):
        countries = await _fetch_rows(
            conn,
            """
            SELECT rc.country_name, rc.country_code, r.code AS region_code
            FROM region_countries rc
            LEFT JOIN regions r ON r.id = rc.region_id
            ORDER BY r.code, rc.country_name
            LIMIT :limit
            """,
            {"limit": limit},
        )

    return {
        "available": True,
        "regions": regions,
        "countries": countries,
    }


async def _build_periods_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False}

    if schema.has_table("periods"):
        payload["available"] = True
        payload["periods"] = await _fetch_rows(
            conn,
            """
            SELECT period_label, period_type, fiscal_year, start_date, end_date
            FROM periods
            ORDER BY end_date DESC
            LIMIT :limit
            """,
            {"limit": limit},
        )

    has_doc_period_fk = schema.has_table("documents") and schema.has_column("documents", "report_period_id")
    has_doc_period_legacy = schema.has_table("documents") and schema.has_column("documents", "report_period")
    if has_doc_period_fk or has_doc_period_legacy:
        if has_doc_period_fk and schema.has_table("periods"):
            period_expr = (
                "COALESCE(p.period_label, d.report_period)"
                if has_doc_period_legacy
                else "p.period_label"
            )
            doc_periods = await _fetch_rows(
                conn,
                f"""
                SELECT {period_expr} AS report_period, COUNT(*) AS document_count
                FROM documents d
                LEFT JOIN periods p ON p.id = d.report_period_id
                WHERE {period_expr} IS NOT NULL AND TRIM({period_expr}) <> ''
                GROUP BY {period_expr}
                ORDER BY report_period DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )
        else:
            doc_periods = await _fetch_rows(
                conn,
                """
                SELECT report_period, COUNT(*) AS document_count
                FROM documents
                WHERE report_period IS NOT NULL AND TRIM(report_period) <> ''
                GROUP BY report_period
                ORDER BY report_period DESC
                LIMIT :limit
                """,
                {"limit": limit},
            )
        payload["available"] = True
        payload["document_periods"] = doc_periods

    if payload.get("available") is False:
        payload["reason"] = "period metadata not found"
    return payload


async def _build_metrics_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    if not schema.has_table("metric_catalog"):
        return {"available": False, "reason": "metric_catalog table not found"}

    select_category = "mc.category" if schema.has_column("metric_catalog", "category") else "NULL"
    select_default_unit = "mc.default_unit" if schema.has_column("metric_catalog", "default_unit") else "NULL"

    rows = await _fetch_rows(
        conn,
        f"""
        SELECT
            mc.id,
            mc.name,
            mc.slug,
            {select_category} AS category,
            {select_default_unit} AS default_unit,
            COUNT(ma.id) AS alias_count
        FROM metric_catalog mc
        LEFT JOIN metric_aliases ma ON ma.metric_id = mc.id
        GROUP BY mc.id, mc.name, mc.slug, {select_category}, {select_default_unit}
        ORDER BY mc.name
        LIMIT :limit
        """,
        {"limit": limit},
    )

    aliases: list[dict[str, Any]] = []
    if schema.has_table("metric_aliases"):
        aliases = await _fetch_rows(
            conn,
            """
            SELECT
                mc.slug AS metric_slug,
                mc.name AS metric_name,
                ma.alias,
                ma.pattern,
                ma.priority,
                ma.unit_override
            FROM metric_aliases ma
            LEFT JOIN metric_catalog mc ON mc.id = ma.metric_id
            ORDER BY mc.name, ma.priority DESC, ma.alias
            LIMIT :limit
            """,
            {"limit": limit},
        )

    definitions: list[dict[str, Any]] = []
    if schema.has_column("metric_catalog", "description") or schema.has_column("metric_catalog", "formula"):
        description_col = "description" if schema.has_column("metric_catalog", "description") else "NULL AS description"
        formula_col = "formula" if schema.has_column("metric_catalog", "formula") else "NULL AS formula"
        definitions = await _fetch_rows(
            conn,
            f"""
            SELECT slug, name, {description_col}, {formula_col}
            FROM metric_catalog
            ORDER BY name
            LIMIT :limit
            """,
            {"limit": limit},
        )

    return {
        "available": True,
        "catalog": rows,
        "aliases": aliases,
        "definitions": definitions,
    }


async def _build_coverage_section(conn, schema: _SchemaSnapshot, limit: int) -> dict[str, Any]:
    payload: dict[str, Any] = {"available": False}

    if schema.has_table("metrics"):
        payload["available"] = True
        payload["metrics_by_catalog"] = await _fetch_rows(
            conn,
            """
            SELECT
                COALESCE(mc.slug, m.name) AS metric_id,
                COALESCE(mc.name, m.name) AS metric_name,
                COUNT(*) AS fact_count,
                COUNT(DISTINCT s.document_id) AS document_count,
                COUNT(DISTINCT m.slide_id) AS slide_count
            FROM metrics m
            LEFT JOIN slides s ON s.id = m.slide_id
            LEFT JOIN metric_catalog mc ON mc.id = m.metric_catalog_id
            GROUP BY COALESCE(mc.slug, m.name), COALESCE(mc.name, m.name)
            ORDER BY fact_count DESC, metric_name
            LIMIT :limit
            """,
            {"limit": limit},
        )

        if schema.has_table("documents") and schema.has_table("clients"):
            payload["metrics_by_client"] = await _fetch_rows(
                conn,
                """
                SELECT COALESCE(c.name, 'Unknown') AS client, COUNT(*) AS metric_fact_count
                FROM metrics m
                LEFT JOIN slides s ON s.id = m.slide_id
                LEFT JOIN documents d ON d.id = s.document_id
                LEFT JOIN clients c ON c.id = d.client_id
                GROUP BY COALESCE(c.name, 'Unknown')
                ORDER BY metric_fact_count DESC, client
                LIMIT :limit
                """,
                {"limit": limit},
            )

        if schema.has_table("regions") and schema.has_table("documents"):
            payload["metrics_by_region"] = await _fetch_rows(
                conn,
                """
                SELECT COALESCE(r.code, 'Unknown') AS region, COUNT(*) AS metric_fact_count
                FROM metrics m
                LEFT JOIN slides s ON s.id = m.slide_id
                LEFT JOIN documents d ON d.id = s.document_id
                LEFT JOIN regions r ON r.id = d.region_id
                GROUP BY COALESCE(r.code, 'Unknown')
                ORDER BY metric_fact_count DESC, region
                LIMIT :limit
                """,
                {"limit": limit},
            )

    if payload.get("available") is False:
        payload["reason"] = "metrics table not found"

    return payload


def _summarize_sections(sections: dict[str, Any]) -> str:
    names = sorted(sections.keys())
    if not names:
        return "No metadata sections were generated."

    if "documents" in sections and isinstance(sections["documents"], dict):
        docs = sections["documents"].get("documents")
        if isinstance(docs, list):
            return (
                f"Metadata catalog ready. Retrieved {len(docs)} document records plus related metadata "
                f"sections: {', '.join(names)}."
            )

    if "metrics" in sections and isinstance(sections["metrics"], dict):
        catalog = sections["metrics"].get("catalog")
        if isinstance(catalog, list):
            return (
                f"Metadata catalog ready. Retrieved {len(catalog)} metric catalog entries and "
                f"sections: {', '.join(names)}."
            )

    return f"Metadata catalog ready. Returned sections: {', '.join(names)}."


@tool(
    desc=(
        "Inspect accessible database metadata and catalog information. Use this for questions like "
        "what documents we have, which clients are available, which metric catalog entries exist, "
        "schema/tables/columns, periods, regions, aliases, and metadata coverage counts."
    ),
    side_effects="read",
    tags=["planner", "metadata", "catalog"],
)
async def metadata_catalog(args: MetadataCatalogArgs, ctx: ToolContext) -> MetadataCatalogResult:
    status = ToolStatusEmitter(ctx, tool_name="metadata_catalog")
    await status.step("Inspecting metadata catalog and access scope.", step_name="Inspect metadata")

    canonical_query = normalize_question_arg(args.question, ctx.tool_context)
    topics = _extract_topics(canonical_query)

    engine_owner = ctx.tool_context.get("metric_query_engine")
    if not isinstance(engine_owner, MetricQueryEngine):
        return MetadataCatalogResult(
            summary_text="Metadata catalog is unavailable because metric_query_engine is missing in tool context.",
            metadata={"topics": sorted(topics), "available": False},
            suggested_queries=_SUGGESTED_METADATA_QUESTIONS,
        )

    metadata_payload: dict[str, Any] = {
        "query": canonical_query,
        "topics": sorted(topics),
        "requested_limit": args.limit,
    }

    async with engine_owner._store.engine.connect() as conn:
        schema = await _inspect_schema(conn)

        sections: dict[str, Any] = {}
        include_all = "overview" in topics
        if include_all or "access" in topics:
            sections["access"] = await _build_access_section(conn, schema, args.limit)
        if include_all or "schema" in topics:
            sections["schema"] = await _build_schema_section(conn, schema, args.limit)
        if include_all or "documents" in topics:
            sections["documents"] = await _build_documents_section(conn, schema, args.limit)
        if include_all or "clients" in topics:
            sections["clients"] = await _build_clients_section(conn, schema, args.limit)
        if include_all or "regions" in topics:
            sections["regions"] = await _build_regions_section(conn, schema, args.limit)
        if include_all or "periods" in topics:
            sections["periods"] = await _build_periods_section(conn, schema, args.limit)
        if include_all or "metrics" in topics:
            sections["metrics"] = await _build_metrics_section(conn, schema, args.limit)
        if include_all or "coverage" in topics:
            sections["coverage"] = await _build_coverage_section(conn, schema, args.limit)

        metadata_payload["sections"] = sections
        metadata_payload["schema_snapshot"] = {
            "dialect": schema.dialect,
            "table_count": len(schema.tables),
            "view_count": len(schema.views),
        }

    summary = _summarize_sections(metadata_payload.get("sections", {}))
    return MetadataCatalogResult(
        summary_text=summary,
        metadata=metadata_payload,
        suggested_queries=_SUGGESTED_METADATA_QUESTIONS,
    )
