"""Database models and utilities."""

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from qbr_intelligence.db.models import (
    Base,
    Chart,
    ChartType,
    Chunk,
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
    MetricTrend,
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

    return engine


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
    "MetricTrend",
    "Chart",
    "ChartType",
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
