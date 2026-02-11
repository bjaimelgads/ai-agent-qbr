"""Database models and utilities."""

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from qbr_intelligence.db.models import (
    Base,
    Chart,
    ChartType,
    Chunk,
    Client,
    Document,
    DocumentFacet,
    DocumentStatus,
    Entity,
    EntityRelation,
    EntityType,
    Facet,
    FacetType,
    FacetValue,
    Image,
    Keyword,
    Metric,
    MetricFactEmbedding,
    MetricCategory,
    MetricCatalog,
    MetricAlias,
    Period,
    Section,
    Slide,
    SlideEntity,
    SlideType,
)


async def init_db(
    database_url: str = "sqlite+aiosqlite:///qbr_intelligence.db",
    echo: bool = False,
) -> AsyncEngine:
    """
    Initialize the database and create all tables.

    Args:
        database_url: Database connection string.
            Examples:
            - "sqlite+aiosqlite:///qbr.db" (SQLite file)
            - "sqlite+aiosqlite:///:memory:" (in-memory SQLite)
            - "postgresql+asyncpg://user:pass@host/db" (PostgreSQL)
        echo: Whether to echo SQL statements

    Returns:
        AsyncEngine instance ready for use

    Example:
        ```python
        engine = await init_db("sqlite+aiosqlite:///qbr.db")
        async with AsyncSession(engine) as session:
            # Use session for queries
        ```
    """
    engine = create_async_engine(database_url, echo=echo)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_metric_columns)
        await conn.run_sync(_ensure_document_columns)
        await conn.run_sync(_backfill_document_period_ids)

    return engine


def _ensure_metric_columns(conn) -> None:
    if conn.engine.dialect.name != "sqlite":
        return
    inspector = inspect(conn)
    if "metrics" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("metrics")}
    needed = {
        "metric_catalog_id": "INTEGER",
        "baseline_text": "TEXT",
        "baseline_type": "TEXT",
        "country": "TEXT",
        "llm_context_label": "TEXT",
    }
    missing = {name: ddl for name, ddl in needed.items() if name not in existing}
    if not missing:
        return
    for name, ddl in missing.items():
        conn.execute(text(f"ALTER TABLE metrics ADD COLUMN {name} {ddl}"))


def _ensure_document_columns(conn) -> None:
    if conn.engine.dialect.name != "sqlite":
        return
    inspector = inspect(conn)
    if "documents" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("documents")}
    needed = {
        "half": "TEXT",
        "client_id": "INTEGER",
        "report_period_id": "INTEGER",
    }
    missing = {name: ddl for name, ddl in needed.items() if name not in existing}
    if not missing:
        return
    for name, ddl in missing.items():
        conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {ddl}"))


def _backfill_document_period_ids(conn) -> None:
    if conn.engine.dialect.name != "sqlite":
        return
    inspector = inspect(conn)
    tables = set(inspector.get_table_names())
    if "documents" not in tables or "periods" not in tables:
        return
    doc_cols = {col["name"] for col in inspector.get_columns("documents")}
    period_cols = {col["name"] for col in inspector.get_columns("periods")}
    if "report_period_id" not in doc_cols or "report_period" not in doc_cols:
        return
    if "id" not in period_cols or "period_label" not in period_cols:
        return

    rows = conn.execute(
        text(
            """
            SELECT DISTINCT TRIM(report_period) AS report_period
            FROM documents
            WHERE report_period_id IS NULL
              AND report_period IS NOT NULL
              AND TRIM(report_period) <> ''
            """
        )
    ).fetchall()

    for row in rows:
        label = row[0]
        if not label:
            continue
        period_row = conn.execute(
            text(
                """
                SELECT id
                FROM periods
                WHERE UPPER(TRIM(period_label)) = UPPER(TRIM(:label))
                ORDER BY id
                LIMIT 1
                """
            ),
            {"label": label},
        ).fetchone()
        period_id = period_row[0] if period_row else None
        if period_id is None:
            conn.execute(
                text("INSERT INTO periods (period_label) VALUES (:label)"),
                {"label": label},
            )
            inserted = conn.execute(text("SELECT last_insert_rowid()")).scalar()
            period_id = int(inserted) if inserted is not None else None
        if period_id is None:
            continue
        conn.execute(
            text(
                """
                UPDATE documents
                SET report_period_id = :period_id
                WHERE report_period_id IS NULL
                  AND report_period IS NOT NULL
                  AND UPPER(TRIM(report_period)) = UPPER(TRIM(:label))
                """
            ),
            {"period_id": period_id, "label": label},
        )


__all__ = [
    # Initialization
    "init_db",
    # Base
    "Base",
    # Models
    "Document",
    "DocumentStatus",
    "Section",
    "Slide",
    "SlideType",
    "Metric",
    "MetricFactEmbedding",
    "MetricCategory",
    "MetricCatalog",
    "MetricAlias",
    "Period",
    "Chart",
    "ChartType",
    "Client",
    "Image",
    "Entity",
    "EntityType",
    "SlideEntity",
    "EntityRelation",
    "Chunk",
    "Facet",
    "FacetType",
    "FacetValue",
    "DocumentFacet",
    "Keyword",
]
