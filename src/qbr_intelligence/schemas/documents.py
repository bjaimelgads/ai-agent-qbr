"""
Pydantic schemas for document models.

These schemas are used for:
- API request/response validation
- Data serialization/deserialization
- Type-safe data transfer between layers
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict


# =============================================================================
# Enums (mirroring DB enums)
# =============================================================================


class DocumentStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    ENHANCING = "enhancing"
    ENHANCED = "enhanced"
    FAILED = "failed"


class SlideType(str, Enum):
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


class MetricCategory(str, Enum):
    PERFORMANCE = "performance"
    COST = "cost"
    REACH = "reach"
    ENGAGEMENT = "engagement"
    REVENUE = "revenue"
    EFFICIENCY = "efficiency"
    OTHER = "other"


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    TABLE = "table"
    COMPARISON = "comparison"
    FUNNEL = "funnel"
    TIMELINE = "timeline"
    KPI_CARD = "kpi_card"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    COMPANY = "company"
    BRAND = "brand"
    PRODUCT = "product"
    PERSON = "person"
    LOCATION = "location"
    MARKET = "market"
    CAMPAIGN = "campaign"
    DATE = "date"
    METRIC_NAME = "metric_name"


# =============================================================================
# Base Schemas
# =============================================================================


class BaseSchema(BaseModel):
    """Base schema with common configuration."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


# =============================================================================
# Document Schemas
# =============================================================================


class DocumentCreate(BaseSchema):
    """Schema for creating a new document."""

    filename: str
    title: str | None = None
    mime_type: str
    client_name: str | None = None
    client_id: int | None = None
    report_period: str | None = None
    fiscal_year: str | None = None
    quarter: str | None = None
    half: str | None = None


class DocumentRead(BaseSchema):
    """Schema for reading document data."""

    id: int
    filename: str
    title: str | None = None
    mime_type: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    processed_at: datetime | None = None

    # Statistics
    page_count: int
    image_count: int
    chunk_count: int

    # LLM-enhanced fields
    executive_summary: str | None = None
    key_wins: list[str] | None = None
    areas_for_improvement: list[str] | None = None
    recommendations: list[str] | None = None
    next_steps: list[str] | None = None

    # Context
    detected_languages: list[str] | None = None
    client_name: str | None = None
    client_id: int | None = None
    report_period: str | None = None
    fiscal_year: str | None = None
    quarter: str | None = None
    half: str | None = None

    # Nested data (optional, loaded on demand)
    sections: list["SectionRead"] | None = None
    slides: list["SlideRead"] | None = None


class DocumentSummary(BaseSchema):
    """Lightweight document summary for listings."""

    id: int
    filename: str
    title: str | None = None
    status: DocumentStatus
    created_at: datetime
    page_count: int
    client_name: str | None = None
    client_id: int | None = None
    report_period: str | None = None
    half: str | None = None


# =============================================================================
# Section Schemas
# =============================================================================


class SectionCreate(BaseSchema):
    """Schema for creating a section."""

    document_id: int
    parent_id: int | None = None
    name: str
    description: str | None = None
    order: int = 0
    start_slide: int | None = None
    end_slide: int | None = None


class SectionRead(BaseSchema):
    """Schema for reading section data."""

    id: int
    document_id: int
    parent_id: int | None = None
    name: str
    description: str | None = None
    order: int
    start_slide: int | None = None
    end_slide: int | None = None
    summary: str | None = None
    key_points: list[str] | None = None

    # Children (for hierarchical structure)
    children: list["SectionRead"] | None = None


# =============================================================================
# Slide Schemas
# =============================================================================


class SlideCreate(BaseSchema):
    """Schema for creating a slide."""

    document_id: int
    section_id: int | None = None
    slide_number: int
    raw_text: str
    speaker_notes: str | None = None
    has_images: bool = False
    image_count: int = 0
    byte_start: int | None = None
    byte_end: int | None = None


class SlideRead(BaseSchema):
    """Schema for reading slide data."""

    id: int
    document_id: int
    section_id: int | None = None
    slide_number: int
    google_slide_id: str | None = None
    raw_text: str
    speaker_notes: str | None = None

    # LLM-enhanced
    slide_type: SlideType = SlideType.UNKNOWN
    title: str | None = None
    key_message: str | None = None
    insights: list[str] | None = None
    action_items: list[str] | None = None

    # Flags
    has_images: bool
    has_charts: bool
    has_tables: bool
    image_count: int

    # Confidence
    classification_confidence: float | None = None

    # Related data (optional)
    metrics: list["MetricRead"] | None = None
    charts: list["ChartRead"] | None = None


# =============================================================================
# Metric Schemas
# =============================================================================


class MetricCreate(BaseSchema):
    """Schema for creating a metric."""

    document_id: int
    slide_id: int | None = None
    raw_value: str
    raw_context: str | None = None
    raw_metric_type: str
    metric_catalog_id: int | None = None
    period_label: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    brand: str | None = None
    baseline_text: str | None = None
    baseline_type: str | None = None
    period_id: int | None = None


class MetricRead(BaseSchema):
    """Schema for reading metric data."""

    id: int
    document_id: int
    slide_id: int | None = None

    # Raw
    raw_value: str
    raw_context: str | None = None
    raw_metric_type: str
    metric_catalog_id: int | None = None

    # Normalized (LLM-enhanced)
    name: str | None = None
    normalized_value: float | None = None
    unit: str | None = None
    category: MetricCategory = MetricCategory.OTHER
    extraction_confidence: float | None = None

    # Period/brand/baseline context
    period_label: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    brand: str | None = None
    baseline_text: str | None = None
    baseline_type: str | None = None
    period_id: int | None = None


# =============================================================================
# Chart Schemas
# =============================================================================


class ChartCreate(BaseSchema):
    """Schema for creating a chart."""

    document_id: int
    slide_id: int | None = None
    chart_type: ChartType = ChartType.UNKNOWN
    detection_confidence: str | None = None
    raw_elements: list[str] | None = None


class ChartRead(BaseSchema):
    """Schema for reading chart data."""

    id: int
    document_id: int
    slide_id: int | None = None
    chart_type: ChartType = ChartType.UNKNOWN
    detection_confidence: str | None = None
    raw_elements: list[str] | None = None

    # Reconstructed (LLM-enhanced)
    title: str | None = None
    x_axis_label: str | None = None
    y_axis_label: str | None = None
    data_series: list[dict] | None = None
    data_table: list[list] | None = None
    insights: list[str] | None = None
    trends_identified: list[str] | None = None


# =============================================================================
# Image Schemas
# =============================================================================


class ImageCreate(BaseSchema):
    """Schema for creating an image record."""

    document_id: int
    slide_id: int | None = None
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    file_path: str | None = None
    image_index: int


class ImageRead(BaseSchema):
    """Schema for reading image data."""

    id: int
    document_id: int
    slide_id: int | None = None
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    file_path: str | None = None
    image_index: int

    # LLM analysis
    image_type: str | None = None
    contains_text: bool = False
    extracted_text: str | None = None
    description: str | None = None
    chart_data: dict | None = None


# =============================================================================
# Entity Schemas
# =============================================================================


class EntityCreate(BaseSchema):
    """Schema for creating an entity."""

    document_id: int
    name: str
    entity_type: EntityType
    normalized_name: str | None = None
    description: str | None = None
    attributes: dict | None = None


class EntityRead(BaseSchema):
    """Schema for reading entity data."""

    id: int
    document_id: int
    name: str
    entity_type: EntityType
    normalized_name: str | None = None
    description: str | None = None
    attributes: dict | None = None
    mention_count: int = 1


# =============================================================================
# Chunk Schemas
# =============================================================================


class ChunkCreate(BaseSchema):
    """Schema for creating a chunk."""

    document_id: int
    content: str
    chunk_index: int
    byte_start: int | None = None
    byte_end: int | None = None
    char_count: int = 0
    start_slide: int | None = None
    end_slide: int | None = None


class ChunkRead(BaseSchema):
    """Schema for reading chunk data."""

    id: int
    document_id: int
    content: str
    chunk_index: int
    byte_start: int | None = None
    byte_end: int | None = None
    char_count: int

    # Embedding
    embedding: list[float] | None = None
    embedding_model: str | None = None

    # LLM-enhanced
    summary: str | None = None
    topics: list[str] | None = None
    importance_score: float | None = None

    # Position
    start_slide: int | None = None
    end_slide: int | None = None


# Forward references for nested models
SectionRead.model_rebuild()
DocumentRead.model_rebuild()
SlideRead.model_rebuild()
