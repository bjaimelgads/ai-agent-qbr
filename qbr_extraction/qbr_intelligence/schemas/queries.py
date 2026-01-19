"""
Pydantic schemas for queries and search.

These schemas define the interface for querying the QBR data,
designed for agent-based access with faceted filtering.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Filter Types
# =============================================================================


class FilterOperator(str, Enum):
    """Operators for filtering."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class SortOrder(str, Enum):
    """Sort order."""

    ASC = "asc"
    DESC = "desc"


# =============================================================================
# Facet Filters
# =============================================================================


class FacetFilter(BaseModel):
    """Filter by facet value."""

    facet_name: str = Field(description="Name of the facet (e.g., 'market', 'campaign')")
    values: list[str] = Field(description="Values to filter by")
    operator: FilterOperator = Field(
        default=FilterOperator.IN, description="Filter operator"
    )


class DocumentFilter(BaseModel):
    """Filter criteria for documents."""

    # Document IDs filter
    document_ids: list[int] | None = Field(default=None, description="Specific document IDs")

    # Status filter (single or list)
    status: str | None = Field(default=None, description="Document status (single)")

    # Date filters (both naming conventions for compatibility)
    date_from: datetime | None = Field(default=None, description="Documents created after")
    date_to: datetime | None = Field(default=None, description="Documents created before")

    # Context filters
    client_name: str | None = Field(default=None)
    fiscal_year: str | None = Field(default=None)
    quarter: str | None = Field(default=None)

    # Facet filters
    facets: list[FacetFilter] | None = Field(
        default=None, description="Facet-based filters"
    )

    # Text search
    search_text: str | None = Field(
        default=None, description="Full-text search query"
    )


class MetricFilter(BaseModel):
    """Filter criteria for metrics."""

    # Category filter (singular for convenience)
    category: str | None = Field(default=None, description="Single category filter")

    # Trend filter
    trends: list[str] | None = Field(default=None)

    # Value filters
    min_value: float | None = Field(default=None)
    max_value: float | None = Field(default=None)

    # Name search
    name_contains: str | None = Field(default=None)

    # Positive/negative trend
    is_positive_trend: bool | None = Field(default=None)


class SlideFilter(BaseModel):
    """Filter criteria for slides."""

    # Type filter (singular for convenience)
    slide_type: str | None = Field(default=None, description="Single slide type filter")

    # Section filter
    section_id: int | None = Field(default=None, description="Filter by section ID")
    section_name: str | None = Field(default=None)

    # Content flags
    has_images: bool | None = Field(default=None)
    has_charts: bool | None = Field(default=None)
    has_action_items: bool | None = Field(default=None)

    # Slide range
    min_slide_number: int | None = Field(default=None)
    max_slide_number: int | None = Field(default=None)


class EntityFilter(BaseModel):
    """Filter criteria for entities."""

    # Type filter (singular for convenience)
    entity_type: str | None = Field(default=None, description="Single entity type filter")

    # Name search
    name_contains: str | None = Field(default=None)

    # Mention count
    min_mentions: int | None = Field(default=None)


# =============================================================================
# Query Request
# =============================================================================


class QueryRequest(BaseModel):
    """
    Universal query request for the QBR data.

    Supports filtering by multiple dimensions and faceted search.
    """

    # Query type (what to query)
    query_type: str = Field(
        description="What to query: 'documents', 'document_detail', 'slides', 'metrics', 'charts', 'entities', 'search', 'facets'"
    )

    # Document scope (optional - limits to specific document)
    document_id: int | None = Field(
        default=None, description="Limit to specific document"
    )

    # Filters
    document_filter: DocumentFilter | None = Field(default=None)
    metric_filter: MetricFilter | None = Field(default=None)
    slide_filter: SlideFilter | None = Field(default=None)
    entity_filter: EntityFilter | None = Field(default=None)

    # Search query (for content search)
    search_query: str | None = Field(
        default=None, description="Text query for content search"
    )

    # Pagination
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)

    # Sorting
    sort_by: str | None = Field(default=None)
    sort_order: SortOrder = Field(default=SortOrder.ASC)

    # Include related data
    include_relationships: bool = Field(
        default=False, description="Include related entities/data"
    )

    # Aggregation
    aggregate_by: str | None = Field(
        default=None, description="Field to aggregate by"
    )


# =============================================================================
# Query Response
# =============================================================================


class SearchResult(BaseModel):
    """A single search result with relevance."""

    id: int = Field(description="ID of the matched record")
    type: str = Field(description="Type of record")
    score: float | None = Field(default=None, description="Relevance score")
    highlight: str | None = Field(
        default=None, description="Highlighted matching text"
    )
    data: dict = Field(description="The matched record data")


class FacetCount(BaseModel):
    """Count for a facet value."""

    value: str = Field(description="Facet value")
    count: int = Field(description="Number of matching records")
    display_name: str | None = Field(default=None)


class FacetResult(BaseModel):
    """Facet results for filtering UI."""

    facet_name: str = Field(description="Name of the facet")
    values: list[FacetCount] = Field(description="Available values with counts")


class AggregationResult(BaseModel):
    """Result of an aggregation query."""

    field: str = Field(description="Field that was aggregated")
    buckets: list[dict] = Field(description="Aggregation buckets")
    total: float | None = Field(default=None, description="Total if applicable")
    average: float | None = Field(default=None, description="Average if applicable")
    min_value: float | None = Field(default=None)
    max_value: float | None = Field(default=None)


class QueryResponse(BaseModel):
    """
    Response from a query request.

    Includes results, facets for refinement, and aggregations.
    """

    # Results (generic list of dicts for flexibility)
    results: list[dict] = Field(default_factory=list, description="Query results")
    total_count: int = Field(default=0, description="Total matching records")

    # Query type echo
    query_type: str = Field(default="", description="The query type that was executed")

    # Error info
    error: str | None = Field(default=None, description="Error message if query failed")

    # Pagination info (optional)
    limit: int | None = Field(default=None)
    offset: int | None = Field(default=None)
    has_more: bool | None = Field(default=None, description="Whether more results are available")

    # Facets for refinement
    available_facets: list[FacetResult] | None = Field(
        default=None, description="Available facets for filtering"
    )

    # Aggregations
    aggregations: list[AggregationResult] | None = Field(
        default=None, description="Aggregation results if requested"
    )

    # Query metadata
    query_time_ms: float | None = Field(
        default=None, description="Query execution time in milliseconds"
    )


# =============================================================================
# Agent Tool Schemas
# =============================================================================


class AgentQueryIntent(str, Enum):
    """Types of queries an agent might make."""

    # Document-level
    LIST_DOCUMENTS = "list_documents"
    GET_DOCUMENT_SUMMARY = "get_document_summary"
    COMPARE_DOCUMENTS = "compare_documents"

    # Metric queries
    GET_METRICS = "get_metrics"
    GET_TOP_PERFORMERS = "get_top_performers"
    GET_UNDERPERFORMERS = "get_underperformers"
    COMPARE_METRICS = "compare_metrics"
    GET_METRIC_TRENDS = "get_metric_trends"

    # Content queries
    GET_SLIDES_BY_TYPE = "get_slides_by_type"
    GET_RECOMMENDATIONS = "get_recommendations"
    GET_ACTION_ITEMS = "get_action_items"
    GET_KEY_INSIGHTS = "get_key_insights"

    # Entity queries
    GET_ENTITIES = "get_entities"
    GET_ENTITY_RELATIONSHIPS = "get_entity_relationships"

    # Faceted queries
    FILTER_BY_MARKET = "filter_by_market"
    FILTER_BY_CAMPAIGN = "filter_by_campaign"
    FILTER_BY_TIME_PERIOD = "filter_by_time_period"

    # Semantic search
    SEMANTIC_SEARCH = "semantic_search"
    FIND_SIMILAR_CONTENT = "find_similar_content"


class AgentToolRequest(BaseModel):
    """
    Request from an agent to query QBR data.

    Designed for natural language → structured query translation.
    """

    intent: AgentQueryIntent = Field(description="What the agent wants to do")
    natural_language_query: str | None = Field(
        default=None, description="Original natural language query"
    )

    # Specific parameters based on intent
    document_id: int | None = Field(default=None)
    metric_names: list[str] | None = Field(default=None)
    entity_names: list[str] | None = Field(default=None)
    markets: list[str] | None = Field(default=None)
    campaigns: list[str] | None = Field(default=None)
    time_period: str | None = Field(default=None)
    slide_types: list[str] | None = Field(default=None)

    # Comparison parameters
    compare_field: str | None = Field(default=None)
    compare_values: list[str] | None = Field(default=None)

    # Limits
    top_n: int = Field(default=10, description="Number of results to return")


class AgentToolResponse(BaseModel):
    """
    Response to an agent tool request.

    Structured for easy consumption by an LLM agent.
    """

    success: bool = Field(description="Whether the query succeeded")
    intent: AgentQueryIntent = Field(description="The intent that was processed")

    # Human-readable summary
    summary: str = Field(
        description="Natural language summary of the results"
    )

    # Structured data
    data: list[dict] = Field(
        default_factory=list, description="Structured result data"
    )

    # Counts and stats
    total_results: int = Field(default=0)
    showing: int = Field(default=0)

    # Additional context
    insights: list[str] | None = Field(
        default=None, description="Key insights from the data"
    )

    # Suggested follow-up queries
    suggested_queries: list[str] | None = Field(
        default=None, description="Suggested follow-up queries"
    )

    # Error info
    error_message: str | None = Field(default=None)
