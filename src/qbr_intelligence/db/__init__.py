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
        "period_label": "TEXT",
        "period_start": "TEXT",
        "period_end": "TEXT",
        "brand": "TEXT",
        "baseline_text": "TEXT",
        "baseline_type": "TEXT",
        "period_id": "INTEGER",
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
    }
    missing = {name: ddl for name, ddl in needed.items() if name not in existing}
    if not missing:
        return
    for name, ddl in missing.items():
        conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {ddl}"))


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
