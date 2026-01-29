"""
Pydantic schemas for LLM structured outputs.

These schemas define the expected output structure from DSPY modules.
They are designed for:
- Structured extraction from LLM responses
- Validation of LLM outputs
- Type-safe integration with the database
"""

from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Slide Analysis Output
# =============================================================================


class SlideTypeOutput(str, Enum):
    """Slide type classification."""

    TITLE = "title"
    AGENDA = "agenda"
    DATA = "data"
    CHART = "chart"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    RECOMMENDATION = "recommendation"
    TRANSITION = "transition"
    APPENDIX = "appendix"
    UNKNOWN = "unknown"


class ExtractedSlideMetric(BaseModel):
    """A metric extracted from slide content."""

    name: str = Field(description="Name of the metric (e.g., 'CTR', 'Reach', 'Cost')")
    value: str = Field(description="The value as it appears (e.g., '28%', '$0.43')")
    normalized_value: float | None = Field(
        default=None, description="Numeric value (e.g., 28.0, 0.43)"
    )
    unit: str | None = Field(
        default=None, description="Unit of measurement (e.g., '%', '$', 'minutes')"
    )
    trend: str | None = Field(
        default=None, description="Trend direction: 'up', 'down', 'stable'"
    )
    context: str | None = Field(
        default=None, description="What this metric relates to"
    )


class SlideAnalysisOutput(BaseModel):
    """
    Structured output from slide analysis.

    Used by DSPY to extract structured insights from each slide.
    """

    slide_type: SlideTypeOutput = Field(
        description="Classification of the slide type"
    )
    title: str | None = Field(
        default=None, description="Extracted or inferred title of the slide"
    )
    key_message: str = Field(
        description="The main message or insight from this slide (1-2 sentences)"
    )
    metrics: list[ExtractedSlideMetric] = Field(
        default_factory=list,
        description="Structured metrics extracted from the slide",
    )
    insights: list[str] = Field(
        default_factory=list,
        description="Key insights or findings from this slide",
    )
    action_items: list[str] = Field(
        default_factory=list,
        description="Recommended actions or next steps mentioned",
    )
    has_chart: bool = Field(
        default=False, description="Whether this slide contains chart data"
    )
    has_comparison: bool = Field(
        default=False, description="Whether this slide compares metrics or options"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the analysis (0-1)"
    )


# =============================================================================
# Metric Normalization Output
# =============================================================================


class MetricCategoryOutput(str, Enum):
    """Category of business metrics."""

    PERFORMANCE = "performance"
    COST = "cost"
    REACH = "reach"
    ENGAGEMENT = "engagement"
    REVENUE = "revenue"
    EFFICIENCY = "efficiency"
    OTHER = "other"


class MetricTrendOutput(str, Enum):
    """Trend direction."""

    UP = "up"
    DOWN = "down"
    STABLE = "stable"
    UNKNOWN = "unknown"


class ExtractedMetric(BaseModel):
    """A single normalized metric."""

    original_value: str = Field(description="Original value as extracted")
    metric_name: str = Field(
        description="Canonical name of the metric (e.g., 'Click-Through Rate')"
    )
    normalized_value: float | None = Field(
        default=None, description="Numeric value extracted"
    )
    unit: str = Field(description="Unit of measurement")
    category: MetricCategoryOutput = Field(description="Category of the metric")
    trend: MetricTrendOutput = Field(
        default=MetricTrendOutput.UNKNOWN, description="Trend direction"
    )
    is_positive: bool | None = Field(
        default=None, description="Whether the trend is positive for business"
    )
    comparison_baseline: str | None = Field(
        default=None, description="What this is being compared to (e.g., 'previous quarter')"
    )
    change_percentage: float | None = Field(
        default=None, description="Percentage change from baseline"
    )
    significance: str = Field(
        description="Business significance of this metric (1 sentence)"
    )
    benchmark_notes: str | None = Field(
        default=None, description="How this compares to industry benchmarks"
    )


class MetricNormalizationOutput(BaseModel):
    """
    Structured output from metric normalization.

    Used by DSPY to normalize and categorize extracted metrics.
    """

    metrics: list[ExtractedMetric] = Field(
        description="List of normalized metrics"
    )
    summary: str = Field(
        description="Brief summary of overall metric performance"
    )
    top_performers: list[str] = Field(
        default_factory=list,
        description="Metrics showing strong positive performance",
    )
    areas_of_concern: list[str] = Field(
        default_factory=list,
        description="Metrics showing concerning trends",
    )


# =============================================================================
# Metric Deduplication Output
# =============================================================================


class MetricDeduplicationOutput(BaseModel):
    """Structured output for metric deduplication."""

    remove_ids: list[str] = Field(
        default_factory=list,
        description="List of metric ids that should be removed as duplicates",
    )
    summary: str = Field(
        default="",
        description="Short explanation of duplicate removal decisions",
    )


# =============================================================================
# Metric Review Output (per-candidate)
# =============================================================================


class MetricReviewResult(BaseModel):
    """Review outcome for a single metric candidate group."""

    chosen_index: int = Field(
        description="Index of the best candidate in the provided list (0-based). Use -1 to keep current."
    )
    normalized_value: float | None = Field(
        default=None, description="Corrected normalized value if needed"
    )
    unit: str | None = Field(default=None, description="Corrected unit if needed")
    notes: str | None = Field(
        default=None, description="Short reasoning or qualifiers"
    )
    confidence: float | None = Field(
        default=None, description="Confidence in the decision (0-1)"
    )
    source_snippet: str | None = Field(
        default=None, description="Exact snippet supporting the choice"
    )


class MetricReviewOutput(BaseModel):
    """Structured output for per-metric review."""

    review: MetricReviewResult = Field(description="Review decision for the metric")


# =============================================================================
# Metric Refinement Output
# =============================================================================


class RefinedMetric(BaseModel):
    """Refined metric extracted from slide context."""

    metric_name: str = Field(description="Canonical metric name from the provided dictionary")
    value: float | None = Field(default=None, description="Primary metric value")
    unit: str | None = Field(default=None, description="Unit of measurement")
    delta_abs: float | None = Field(
        default=None, description="Absolute change value if present (e.g., $0.83 cheaper)"
    )
    delta_pct: float | None = Field(
        default=None, description="Percent change if present (e.g., -32%)"
    )
    baseline_text: str | None = Field(
        default=None, description="Baseline description (e.g., vs last half)"
    )
    notes: str | None = Field(
        default=None, description="Short explanation or qualifiers"
    )
    confidence: float | None = Field(
        default=None, description="Confidence in the refinement (0-1)"
    )
    source_ids: list[str] = Field(
        default_factory=list,
        description="IDs of the candidate metrics used to derive this metric",
    )
    source_snippet: str | None = Field(
        default=None,
        description="Exact text snippet the value was taken from",
    )


class MetricRefinementOutput(BaseModel):
    """Structured output for per-slide metric refinement."""

    metrics: list[RefinedMetric] = Field(
        default_factory=list,
        description="Refined metrics for the slide",
    )


# =============================================================================
# Metric Context Output
# =============================================================================


class MetricContextItem(BaseModel):
    """Context fields for a metric (period/brand/baseline)."""

    metric_id: str = Field(description="Metric candidate ID")
    period_label: str | None = Field(
        default=None, description="Period label (e.g., H2 FY25, Q1 2025)"
    )
    period_start: str | None = Field(
        default=None, description="Period start date (ISO-8601), if available"
    )
    period_end: str | None = Field(
        default=None, description="Period end date (ISO-8601), if available"
    )
    brand: str | None = Field(
        default=None, description="Brand or client associated with the metric"
    )
    baseline_text: str | None = Field(
        default=None, description="Baseline text (e.g., vs last quarter)"
    )
    baseline_type: str | None = Field(
        default=None, description="Baseline type (e.g., yoy, qoq, mom, target)"
    )
    source_snippet: str | None = Field(
        default=None, description="Text snippet supporting the context"
    )
    confidence: float | None = Field(
        default=None, description="Confidence for the context extraction (0-1)"
    )


class MetricContextOutput(BaseModel):
    """Structured output for per-slide metric context."""

    contexts: list[MetricContextItem] = Field(
        default_factory=list,
        description="Context items for metrics on the slide",
    )


# =============================================================================
# Chart Reconstruction Output
# =============================================================================


class ChartTypeOutput(str, Enum):
    """Types of charts."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    TABLE = "table"
    COMPARISON = "comparison"
    FUNNEL = "funnel"
    TIMELINE = "timeline"
    KPI_CARD = "kpi_card"
    UNKNOWN = "unknown"


class DataSeriesItem(BaseModel):
    """A data series in a chart."""

    label: str = Field(description="Label for this data series")
    values: list[float | str] = Field(description="Values in this series")
    unit: str | None = Field(default=None, description="Unit of measurement")


class ChartReconstructionOutput(BaseModel):
    """
    Structured output from chart reconstruction.

    Used by DSPY to rebuild chart data from text elements.
    """

    chart_type: ChartTypeOutput = Field(description="Type of chart detected")
    title: str | None = Field(default=None, description="Title of the chart")
    x_axis_label: str | None = Field(default=None, description="X-axis label")
    y_axis_label: str | None = Field(default=None, description="Y-axis label")
    data_series: list[DataSeriesItem] = Field(
        default_factory=list, description="Data series in the chart"
    )
    data_table: list[list[str]] = Field(
        default_factory=list,
        description="Data as a table (first row = headers)",
    )
    insights: list[str] = Field(
        default_factory=list,
        description="Key insights from the visualization",
    )
    trends: list[str] = Field(
        default_factory=list, description="Trends identified in the data"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in reconstruction (0-1)"
    )


# =============================================================================
# Executive Summary Output
# =============================================================================


class ExecutiveSummaryOutput(BaseModel):
    """
    Structured output for executive summary generation.

    Used by DSPY to generate a comprehensive summary of the QBR.
    """

    summary: str = Field(
        description="Executive summary (2-3 paragraphs)"
    )
    key_wins: list[str] = Field(
        description="Major achievements and successes (3-5 items)"
    )
    areas_for_improvement: list[str] = Field(
        description="Areas needing attention (2-4 items)"
    )
    recommendations: list[str] = Field(
        description="Strategic recommendations (3-5 items)"
    )
    next_steps: list[str] = Field(
        description="Concrete next steps (3-5 items)"
    )
    overall_sentiment: str = Field(
        description="Overall sentiment: 'positive', 'neutral', 'negative', 'mixed'"
    )
    key_metrics_highlighted: list[str] = Field(
        default_factory=list,
        description="Most important metrics from the report",
    )
    time_period_covered: str | None = Field(
        default=None, description="Time period this report covers"
    )


# =============================================================================
# Entity Extraction Output
# =============================================================================


class EntityTypeOutput(str, Enum):
    """Types of named entities."""

    COMPANY = "company"
    BRAND = "brand"
    PRODUCT = "product"
    PERSON = "person"
    LOCATION = "location"
    MARKET = "market"
    CAMPAIGN = "campaign"
    DATE = "date"
    METRIC_NAME = "metric_name"


class ExtractedEntity(BaseModel):
    """A single extracted entity."""

    name: str = Field(description="Name of the entity")
    entity_type: EntityTypeOutput = Field(description="Type of entity")
    normalized_name: str | None = Field(
        default=None, description="Canonical/normalized form of the name"
    )
    description: str | None = Field(
        default=None, description="Brief description or context"
    )
    mentions: int = Field(default=1, description="Number of times mentioned")
    related_entities: list[str] = Field(
        default_factory=list,
        description="Names of related entities",
    )


class EntityRelationOutput(BaseModel):
    """A relationship between entities."""

    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(
        description="Type of relationship (e.g., 'owns', 'competes_with', 'markets_in')"
    )
    description: str | None = Field(
        default=None, description="Description of the relationship"
    )


class EntityExtractionOutput(BaseModel):
    """
    Structured output from entity extraction.

    Used by DSPY to extract named entities and relationships.
    """

    entities: list[ExtractedEntity] = Field(
        description="All extracted entities"
    )
    relationships: list[EntityRelationOutput] = Field(
        default_factory=list,
        description="Relationships between entities",
    )
    companies: list[str] = Field(
        default_factory=list, description="Company/brand names"
    )
    products: list[str] = Field(
        default_factory=list, description="Product/service names"
    )
    people: list[str] = Field(
        default_factory=list, description="Person names"
    )
    markets: list[str] = Field(
        default_factory=list, description="Markets/regions"
    )
    campaigns: list[str] = Field(
        default_factory=list, description="Campaign names"
    )
    dates: list[str] = Field(
        default_factory=list, description="Date/time references"
    )


# =============================================================================
# Image Analysis Output
# =============================================================================


class ImageTypeOutput(str, Enum):
    """Types of images."""

    CHART = "chart"
    PHOTO = "photo"
    LOGO = "logo"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    ICON = "icon"
    DECORATIVE = "decorative"
    UNKNOWN = "unknown"


class ImageAnalysisOutput(BaseModel):
    """
    Structured output from image analysis.

    Used by DSPY to analyze images for content and data.
    """

    image_type: ImageTypeOutput = Field(description="Type of image")
    contains_text: bool = Field(description="Whether the image contains text")
    extracted_text: str | None = Field(
        default=None, description="Text extracted from the image (OCR)"
    )
    description: str = Field(
        description="Accessibility description of the image"
    )
    is_chart: bool = Field(
        default=False, description="Whether this is a chart/graph"
    )
    chart_data: ChartReconstructionOutput | None = Field(
        default=None, description="Reconstructed chart data if applicable"
    )
    brand_elements: list[str] = Field(
        default_factory=list,
        description="Brand elements identified (logos, colors, etc.)",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in analysis (0-1)"
    )
