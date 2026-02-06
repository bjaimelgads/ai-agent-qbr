"""
QBR Intelligence - Document Intelligence for Quarterly Business Reviews.

A comprehensive framework for extracting, enhancing, and querying QBR documents
using kreuzberg for extraction, SQLAlchemy for persistence, DSPY for LLM
enhancement, and faceted search for agent queries.

Main Components:
- db: SQLAlchemy models and database utilities
- schemas: Pydantic models for validation and serialization
- llm: DSPY modules for LLM-powered enhancement
- pipeline: Document processing pipeline
- query: Agent-queryable interface with faceted search

Quick Start:
    ```python
    from qbr_intelligence import QBRProcessor, QBRQueryInterface, init_db

    # Initialize database
    engine = await init_db("sqlite+aiosqlite:///qbr.db")

    # Process a document
    processor = QBRProcessor(engine)
    document = await processor.process_document("qbr.pptx")

    # Query the data
    async with AsyncSession(engine) as session:
        interface = QBRQueryInterface(session)
        summary = await interface.get_document_summary(document.id)
    ```
"""

import os

__version__ = "0.1.0"

_LIGHT_IMPORT = os.getenv("QBR_INTELLIGENCE_LIGHT_IMPORT") == "1"

if not _LIGHT_IMPORT:
    # Database
    from qbr_intelligence.db import init_db
    from qbr_intelligence.db.models import (
        Chart,
        Chunk,
        Document,
        DocumentFacet,
        DocumentStatus,
        Entity,
        EntityRelation,
        EntityType,
        Facet,
        FacetValue,
        Image,
        Keyword,
        Metric,
        MetricCategory,
        Period,
        Section,
        Slide,
        SlideEntity,
        SlideType,
    )

    # Pipeline
    from qbr_intelligence.pipeline import QBRProcessor

    # LLM Modules
    from qbr_intelligence.llm import (
        ChartReconstructor,
        EntityExtractor,
        ExecutiveSummarizer,
        MetricNormalizer,
        QBREnhancementPipeline,
        SlideAnalyzer,
    )

    # Query Interface
    from qbr_intelligence.query import (
        QBRQueryInterface,
        get_action_items,
        get_document_summary,
        get_key_insights,
        get_metrics_by_category,
        get_recommendations,
        get_slides_by_type,
        get_top_metrics,
        list_documents,
        search_content,
    )

    __all__ = [
        # Version
        "__version__",
        # Database init
        "init_db",
        # Database models
        "Document",
        "DocumentStatus",
        "Section",
        "Slide",
        "SlideType",
        "Metric",
        "MetricCategory",
        "Period",
        "Chart",
        "Image",
        "Entity",
        "EntityType",
        "EntityRelation",
        "SlideEntity",
        "Chunk",
        "Facet",
        "FacetValue",
        "DocumentFacet",
        "Keyword",
        # Pipeline
        "QBRProcessor",
        # LLM Modules
        "SlideAnalyzer",
        "MetricNormalizer",
        "ChartReconstructor",
        "ExecutiveSummarizer",
        "EntityExtractor",
        "QBREnhancementPipeline",
        # Query Interface
        "QBRQueryInterface",
        "list_documents",
        "get_document_summary",
        "search_content",
        "get_slides_by_type",
        "get_metrics_by_category",
        "get_top_metrics",
        "get_key_insights",
        "get_recommendations",
        "get_action_items",
    ]
else:
    __all__ = ["__version__"]
