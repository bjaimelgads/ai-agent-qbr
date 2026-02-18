"""
Agent tool functions for QBR data queries.

These functions provide a simple interface for AI agents to query QBR data.
Each function is designed to be called as a tool by an agent, with clear
input parameters and structured output.
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from qbr_intelligence.db.models import (
    DocumentStatus,
    EntityType,
    MetricCategory,
)
from qbr_intelligence.query.interface import QBRQueryInterface


# =============================================================================
# Document Tools
# =============================================================================


async def list_documents(
    session: AsyncSession,
    status: str | None = None,
    client_name: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    List all QBR documents with optional filtering.

    Use this tool to:
    - Get an overview of available QBR documents
    - Filter documents by client name or status
    - Find specific documents to analyze

    Args:
        session: Database session
        status: Filter by status (pending, processing, enhanced, failed)
        client_name: Filter by client name (partial match)
        limit: Maximum number of documents to return

    Returns:
        Dictionary with:
        - documents: List of document summaries
        - count: Number of documents returned

    Example agent usage:
        "List all QBR documents for Disney"
        "Show me pending documents"
        "What QBR documents do we have?"
    """
    interface = QBRQueryInterface(session)

    status_enum = None
    if status:
        try:
            status_enum = DocumentStatus(status)
        except ValueError:
            return {"error": f"Invalid status: {status}. Valid values: pending, processing, enhanced, failed"}

    documents = await interface.list_documents(
        status=status_enum,
        client_name=client_name,
        limit=limit,
    )

    return {
        "documents": documents,
        "count": len(documents),
    }


async def get_document_summary(
    session: AsyncSession,
    document_id: int,
) -> dict[str, Any]:
    """
    Get the executive summary and key insights for a document.

    Use this tool to:
    - Get a high-level overview of a QBR
    - Find key wins and areas for improvement
    - Get recommendations and action items

    Args:
        session: Database session
        document_id: ID of the document

    Returns:
        Dictionary with:
        - executive_summary: Overall summary
        - key_wins: List of achievements
        - areas_for_improvement: List of areas needing work
        - recommendations: Strategic recommendations
        - top_metrics: Most significant metrics
        - action_items: Concrete next steps

    Example agent usage:
        "What are the key takeaways from document 1?"
        "Summarize the Disney QBR"
        "What are the main recommendations?"
    """
    interface = QBRQueryInterface(session)
    summary = await interface.get_document_summary(document_id)

    if not summary:
        return {"error": f"Document {document_id} not found"}

    return summary


# =============================================================================
# Metric Tools
# =============================================================================


async def get_metrics_by_category(
    session: AsyncSession,
    document_id: int,
) -> dict[str, Any]:
    """
    Get all metrics from a document organized by category.

    Use this tool to:
    - Understand performance across different areas
    - Compare metrics within categories
    - Find metrics in a specific business area

    Args:
        session: Database session
        document_id: ID of the document

    Returns:
        Dictionary mapping category names to lists of metrics.
        Categories include: revenue, growth, engagement, performance,
        customer, cost, market_share, conversion, efficiency, other

    Example agent usage:
        "What are all the revenue metrics in document 1?"
        "Show me engagement metrics from the Disney QBR"
        "Break down all metrics by category"
    """
    interface = QBRQueryInterface(session)
    metrics = await interface.get_metrics_by_category(document_id)

    return {
        "document_id": document_id,
        "categories": metrics,
        "total_metrics": sum(len(m) for m in metrics.values()),
    }


async def get_top_metrics(
    session: AsyncSession,
    document_id: int,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Get the most significant metrics from a document.

    Use this tool to:
    - Identify the most important KPIs
    - Find metrics that matter most
    - Get a quick performance snapshot

    Args:
        session: Database session
        document_id: ID of the document
        limit: Number of top metrics to return

    Returns:
        List of the most significant metrics with their values,
        trends, and context.

    Example agent usage:
        "What are the top 5 KPIs in this QBR?"
        "Show me the most important metrics"
        "What metrics should I focus on?"
    """
    interface = QBRQueryInterface(session)
    metrics = await interface.get_top_metrics(document_id, limit=limit)

    return {
        "document_id": document_id,
        "top_metrics": metrics,
        "count": len(metrics),
    }


async def search_metrics(
    session: AsyncSession,
    name_contains: str,
    document_id: int | None = None,
    category: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Search for specific metrics by name or category.

    Use this tool to:
    - Find specific metrics across documents
    - Filter metrics by category
    - Search for metrics matching a pattern

    Args:
        session: Database session
        name_contains: Text to search for in metric names
        document_id: Optional document filter
        category: Optional category filter
        limit: Maximum results

    Returns:
        List of matching metrics with full details.

    Example agent usage:
        "Find all revenue-related metrics"
        "Search for CTR metrics in document 1"
        "What metrics mention 'conversion'?"
    """
    interface = QBRQueryInterface(session)

    category_enum = None
    if category:
        try:
            category_enum = MetricCategory(category)
        except ValueError:
            pass  # Ignore invalid category

    metrics = await interface.get_metrics(
        document_id=document_id,
        category=category_enum,
        name_contains=name_contains,
        limit=limit,
    )

    return {
        "metrics": metrics,
        "count": len(metrics),
        "search_query": name_contains,
    }


# =============================================================================
# Slide Tools
# =============================================================================


async def get_slides_by_type(
    session: AsyncSession,
    document_id: int,
) -> dict[str, Any]:
    """
    Get all slides from a document grouped by type.

    Use this tool to:
    - Find specific types of slides (performance, recommendations, etc.)
    - Navigate the document structure
    - Understand the document composition

    Args:
        session: Database session
        document_id: ID of the document

    Returns:
        Dictionary mapping slide types to lists of slides.
        Types include: title, agenda, overview, performance,
        analysis, recommendations, action_items, summary, etc.

    Example agent usage:
        "Show me all recommendation slides"
        "Find the performance slides in document 1"
        "What types of slides are in this QBR?"
    """
    interface = QBRQueryInterface(session)
    slides = await interface.get_slides_by_type(document_id)

    return {
        "document_id": document_id,
        "slide_types": slides,
        "total_slides": sum(len(s) for s in slides.values()),
    }


async def get_slide_content(
    session: AsyncSession,
    slide_id: int,
) -> dict[str, Any]:
    """
    Get full content of a specific slide.

    Use this tool to:
    - Read the complete text of a slide
    - See charts and images on a slide
    - Get detailed analysis of a specific slide

    Args:
        session: Database session
        slide_id: ID of the slide

    Returns:
        Full slide data including text, charts, entities,
        key messages, and action items.

    Example agent usage:
        "What's on slide 15?"
        "Show me the content of slide ID 42"
        "Read the executive summary slide"
    """
    interface = QBRQueryInterface(session)
    slide = await interface.get_slide_detail(slide_id)

    if not slide:
        return {"error": f"Slide {slide_id} not found"}

    return slide


# =============================================================================
# Recommendation and Action Item Tools
# =============================================================================


async def get_recommendations(
    session: AsyncSession,
    document_id: int,
) -> dict[str, Any]:
    """
    Get all recommendations from a document.

    Use this tool to:
    - Find strategic recommendations
    - Get advice for improvement
    - Understand suggested next steps

    Args:
        session: Database session
        document_id: ID of the document

    Returns:
        Dictionary with document-level and slide-level recommendations.

    Example agent usage:
        "What are the recommendations in this QBR?"
        "What does the document suggest we do?"
        "Show me the strategic advice"
    """
    interface = QBRQueryInterface(session)
    summary = await interface.get_document_summary(document_id)

    if not summary:
        return {"error": f"Document {document_id} not found"}

    return {
        "document_id": document_id,
        "recommendations": summary.get("recommendations"),
        "areas_for_improvement": summary.get("areas_for_improvement"),
    }


async def get_action_items(
    session: AsyncSession,
    document_id: int,
) -> dict[str, Any]:
    """
    Get all action items from a document.

    Use this tool to:
    - Find concrete next steps
    - Create task lists from the QBR
    - Identify what needs to be done

    Args:
        session: Database session
        document_id: ID of the document

    Returns:
        List of action items with their source slides.

    Example agent usage:
        "What action items came out of this QBR?"
        "What tasks need to be done?"
        "Create a to-do list from the recommendations"
    """
    interface = QBRQueryInterface(session)
    summary = await interface.get_document_summary(document_id)

    if not summary:
        return {"error": f"Document {document_id} not found"}

    return {
        "document_id": document_id,
        "action_items": summary.get("action_items", []),
        "count": len(summary.get("action_items", [])),
    }


async def get_key_insights(
    session: AsyncSession,
    document_id: int,
) -> dict[str, Any]:
    """
    Get aggregated insights from all slides in a document.

    Use this tool to:
    - Get all insights in one place
    - Understand key findings across slides
    - Get a comprehensive analysis summary

    Args:
        session: Database session
        document_id: ID of the document

    Returns:
        Dictionary with insights, key wins, and areas for improvement.

    Example agent usage:
        "What are the key insights from this QBR?"
        "Summarize the main findings"
        "What did we learn from this review?"
    """
    interface = QBRQueryInterface(session)
    insights = await interface.get_insights_summary(document_id)

    return insights


# =============================================================================
# Entity Tools
# =============================================================================


async def get_entities(
    session: AsyncSession,
    document_id: int,
    entity_type: str | None = None,
) -> dict[str, Any]:
    """
    Get entities (companies, people, products, etc.) from a document.

    Use this tool to:
    - Find all companies mentioned
    - Identify key people referenced
    - List products or campaigns discussed
    - Find markets or regions mentioned

    Args:
        session: Database session
        document_id: ID of the document
        entity_type: Optional filter (company, person, product, market, campaign, date)

    Returns:
        List of entities with their types and mention counts.

    Example agent usage:
        "What companies are mentioned in this QBR?"
        "Who are the key people referenced?"
        "What campaigns are discussed?"
    """
    interface = QBRQueryInterface(session)

    type_enum = None
    if entity_type:
        try:
            type_enum = EntityType(entity_type)
        except ValueError:
            return {"error": f"Invalid entity type: {entity_type}. Valid values: company, person, product, market, campaign, date, other"}

    entities = await interface.get_entities(
        document_id=document_id,
        entity_type=type_enum,
    )

    return {
        "document_id": document_id,
        "entities": entities,
        "count": len(entities),
    }


async def search_entities(
    session: AsyncSession,
    name_contains: str,
    document_id: int | None = None,
) -> dict[str, Any]:
    """
    Search for entities by name across documents.

    Use this tool to:
    - Find all mentions of a company
    - Search for a specific person
    - Find documents mentioning a product

    Args:
        session: Database session
        name_contains: Text to search for in entity names
        document_id: Optional document filter

    Returns:
        List of matching entities.

    Example agent usage:
        "Find all mentions of Disney"
        "Search for 'streaming' in entities"
        "What documents mention Disney+"
    """
    interface = QBRQueryInterface(session)
    entities = await interface.get_entities(
        document_id=document_id,
        name_contains=name_contains,
    )

    return {
        "entities": entities,
        "count": len(entities),
        "search_query": name_contains,
    }


# =============================================================================
# Search Tools
# =============================================================================


async def search_content(
    session: AsyncSession,
    query: str,
    document_id: int | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Search for content within documents.

    Use this tool to:
    - Find specific information in documents
    - Search across all QBR content
    - Locate mentions of specific topics

    Args:
        session: Database session
        query: Search query text
        document_id: Optional document filter
        limit: Maximum results

    Returns:
        List of matching content chunks with context.

    Example agent usage:
        "Search for 'streaming revenue' in document 1"
        "Find mentions of Q4 performance"
        "Search all documents for 'subscriber growth'"
    """
    interface = QBRQueryInterface(session)
    results = await interface.search_content(
        query_text=query,
        document_id=document_id,
        limit=limit,
    )

    return {
        "results": results,
        "count": len(results),
        "query": query,
    }


# =============================================================================
# Chart Tools
# =============================================================================


async def get_charts(
    session: AsyncSession,
    document_id: int,
) -> dict[str, Any]:
    """
    Get all charts from a document with their data.

    Use this tool to:
    - Find data visualizations in the document
    - Get chart data for analysis
    - Understand trends from charts

    Args:
        session: Database session
        document_id: ID of the document

    Returns:
        List of charts with their types, data, and insights.

    Example agent usage:
        "What charts are in this QBR?"
        "Show me the data from charts in document 1"
        "Find the revenue trend chart"
    """
    interface = QBRQueryInterface(session)
    charts = await interface.get_charts_summary(document_id)

    return {
        "document_id": document_id,
        "charts": charts,
        "count": len(charts),
    }


# =============================================================================
# Facet Tools
# =============================================================================


async def get_available_facets(
    session: AsyncSession,
) -> dict[str, Any]:
    """
    Get all available facets for filtering documents.

    Use this tool to:
    - Understand filtering options
    - See what dimensions are available
    - Plan faceted searches

    Args:
        session: Database session

    Returns:
        List of facets with their possible values.

    Example agent usage:
        "What filtering options are available?"
        "How can I filter documents?"
        "What facets can I use?"
    """
    interface = QBRQueryInterface(session)
    facets = await interface.get_available_facets()

    return {
        "facets": facets,
        "count": len(facets),
    }


async def filter_by_facets(
    session: AsyncSession,
    facet_filters: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Filter documents by facet values.

    Use this tool to:
    - Find documents matching specific criteria
    - Filter by market, campaign, time period, etc.
    - Narrow down document selection

    Args:
        session: Database session
        facet_filters: List of {facet_name: str, values: list[str]}

    Returns:
        List of documents matching the facet criteria.

    Example agent usage:
        "Find all QBRs for the US market"
        "Show documents from Q4 2024"
        "Filter for streaming-related QBRs"
    """
    interface = QBRQueryInterface(session)
    documents = await interface.filter_documents_by_facets(facet_filters)

    return {
        "documents": documents,
        "count": len(documents),
        "filters_applied": facet_filters,
    }


# =============================================================================
# Comparison Tools
# =============================================================================


async def compare_metric_across_documents(
    session: AsyncSession,
    metric_name: str,
    document_ids: list[int] | None = None,
) -> dict[str, Any]:
    """
    Compare a specific metric across multiple documents.

    Use this tool to:
    - Track metric trends over time
    - Compare performance across QBRs
    - Identify metric changes

    Args:
        session: Database session
        metric_name: Name of the metric to compare
        document_ids: Optional list of specific documents to compare

    Returns:
        List of metric values across documents with context.

    Example agent usage:
        "Compare revenue growth across all QBRs"
        "How has subscriber count changed over time?"
        "Track CTR across documents 1, 2, and 3"
    """
    interface = QBRQueryInterface(session)
    comparisons = await interface.compare_metrics_across_documents(
        metric_name=metric_name,
        document_ids=document_ids,
    )

    return {
        "metric_name": metric_name,
        "comparisons": comparisons,
        "document_count": len(comparisons),
    }


# =============================================================================
# Tool Registry for Agents
# =============================================================================


AGENT_TOOLS = {
    # Document tools
    "list_documents": {
        "function": list_documents,
        "description": "List all QBR documents with optional filtering by status or client name",
        "parameters": ["status", "client_name", "limit"],
    },
    "get_document_summary": {
        "function": get_document_summary,
        "description": "Get executive summary, key wins, recommendations, and action items for a document",
        "parameters": ["document_id"],
    },
    # Metric tools
    "get_metrics_by_category": {
        "function": get_metrics_by_category,
        "description": "Get all metrics organized by category (revenue, growth, engagement, etc.)",
        "parameters": ["document_id"],
    },
    "get_top_metrics": {
        "function": get_top_metrics,
        "description": "Get the most significant KPIs from a document",
        "parameters": ["document_id", "limit"],
    },
    "search_metrics": {
        "function": search_metrics,
        "description": "Search for metrics by name or filter by category",
        "parameters": ["name_contains", "document_id", "category", "limit"],
    },
    # Slide tools
    "get_slides_by_type": {
        "function": get_slides_by_type,
        "description": "Get slides organized by type (performance, recommendations, etc.)",
        "parameters": ["document_id"],
    },
    "get_slide_content": {
        "function": get_slide_content,
        "description": "Get full content of a specific slide including charts and entities",
        "parameters": ["slide_id"],
    },
    # Insight tools
    "get_recommendations": {
        "function": get_recommendations,
        "description": "Get strategic recommendations and areas for improvement",
        "parameters": ["document_id"],
    },
    "get_action_items": {
        "function": get_action_items,
        "description": "Get concrete action items and next steps",
        "parameters": ["document_id"],
    },
    "get_key_insights": {
        "function": get_key_insights,
        "description": "Get aggregated insights, key wins, and areas for improvement",
        "parameters": ["document_id"],
    },
    # Entity tools
    "get_entities": {
        "function": get_entities,
        "description": "Get entities (companies, people, products, markets) from a document",
        "parameters": ["document_id", "entity_type"],
    },
    "search_entities": {
        "function": search_entities,
        "description": "Search for entities by name across documents",
        "parameters": ["name_contains", "document_id"],
    },
    # Search tools
    "search_content": {
        "function": search_content,
        "description": "Full-text search within document content",
        "parameters": ["query", "document_id", "limit"],
    },
    # Chart tools
    "get_charts": {
        "function": get_charts,
        "description": "Get all charts from a document with their reconstructed data",
        "parameters": ["document_id"],
    },
    # Facet tools
    "get_available_facets": {
        "function": get_available_facets,
        "description": "Get all available facets for filtering documents",
        "parameters": [],
    },
    "filter_by_facets": {
        "function": filter_by_facets,
        "description": "Filter documents by facet values (market, campaign, time period)",
        "parameters": ["facet_filters"],
    },
    # Comparison tools
    "compare_metric_across_documents": {
        "function": compare_metric_across_documents,
        "description": "Compare a metric across multiple QBR documents",
        "parameters": ["metric_name", "document_ids"],
    },
}


def get_tool_descriptions() -> str:
    """
    Get formatted descriptions of all available tools.

    Useful for providing tool documentation to agents.
    """
    lines = ["Available QBR Query Tools:", "=" * 50]

    for name, info in AGENT_TOOLS.items():
        lines.append(f"\n{name}:")
        lines.append(f"  {info['description']}")
        if info["parameters"]:
            lines.append(f"  Parameters: {', '.join(info['parameters'])}")

    return "\n".join(lines)
