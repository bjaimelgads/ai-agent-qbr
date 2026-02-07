"""
SQLAlchemy ORM models for QBR Intelligence.

Designed for:
- Hierarchical document structure (Document → Sections → Slides)
- Faceted search and filtering
- Entity relationships and knowledge graph
- LLM-enhanced metadata storage
- Agent-queryable interface
"""

from datetime import datetime
from enum import Enum

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


# =============================================================================
# Enums
# =============================================================================


class DocumentStatus(str, Enum):
    """Processing status of a document."""

    PENDING = "pending"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    ENHANCING = "enhancing"
    ENHANCED = "enhanced"
    FAILED = "failed"


class Client(Base):
    """Client/advertiser registry."""

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)

    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="client"
    )


class SlideType(str, Enum):
    """Type classification for slides."""

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
    """Category of business metrics."""

    PERFORMANCE = "performance"  # CTR, conversion rate, etc.
    COST = "cost"  # CPC, CPM, spend
    REACH = "reach"  # impressions, unique reach
    ENGAGEMENT = "engagement"  # time spent, interactions
    REVENUE = "revenue"  # ROAS, revenue
    EFFICIENCY = "efficiency"  # CPPC, cost per conversion
    OTHER = "other"


class EntityType(str, Enum):
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


class ChartType(str, Enum):
    """Types of charts/visualizations."""

    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    TABLE = "table"
    COMPARISON = "comparison"
    FUNNEL = "funnel"
    TIMELINE = "timeline"
    KPI_CARD = "kpi_card"
    UNKNOWN = "unknown"


class FacetType(str, Enum):
    """Types of facets for filtering."""

    TIME_PERIOD = "time_period"
    MARKET = "market"
    CAMPAIGN = "campaign"
    METRIC_TYPE = "metric_type"
    SECTION = "section"
    BRAND = "brand"
    PRODUCT = "product"
    CUSTOM = "custom"


# =============================================================================
# Region Models
# =============================================================================


class Region(Base):
    """Geographic region (e.g., US, EMEA)."""

    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)

    countries: Mapped[list["RegionCountry"]] = relationship(
        "RegionCountry", back_populates="region", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="region"
    )
    metrics: Mapped[list["Metric"]] = relationship("Metric", back_populates="region")


class RegionCountry(Base):
    """Country membership for a region (canonical names)."""

    __tablename__ = "region_countries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    region_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("regions.id", ondelete="CASCADE"), nullable=False
    )
    country_name: Mapped[str] = mapped_column(String(100), nullable=False)
    country_code: Mapped[str | None] = mapped_column(String(10), nullable=True)

    region: Mapped["Region"] = relationship("Region", back_populates="countries")


# =============================================================================
# Document Model
# =============================================================================


class Document(Base):
    """
    Root document representing a QBR presentation.

    Contains metadata about the entire document and serves as the
    root of the document hierarchy.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Processing metadata
    status: Mapped[str] = mapped_column(
        String(50), default=DocumentStatus.PENDING.value
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Document statistics
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    slide_count: Mapped[int] = mapped_column(Integer, default=0)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)

    # Raw extraction metadata (JSON blob)
    extraction_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # LLM-enhanced fields
    executive_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_wins: Mapped[list | None] = mapped_column(JSON, nullable=True)
    areas_for_improvement: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recommendations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    next_steps: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Detected languages
    detected_languages: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Client/context info (extracted or provided)
    client_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("clients.id", ondelete="SET NULL"), nullable=True
    )
    region_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("regions.id", ondelete="SET NULL"), nullable=True
    )
    report_period: Mapped[str | None] = mapped_column(String(100), nullable=True)
    fiscal_year: Mapped[str | None] = mapped_column(String(20), nullable=True)
    quarter: Mapped[str | None] = mapped_column(String(10), nullable=True)
    half: Mapped[str | None] = mapped_column(String(10), nullable=True)

    # Relationships
    sections: Mapped[list["Section"]] = relationship(
        "Section", back_populates="document", cascade="all, delete-orphan"
    )
    slides: Mapped[list["Slide"]] = relationship(
        "Slide", back_populates="document", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", back_populates="document", cascade="all, delete-orphan"
    )
    region: Mapped["Region | None"] = relationship(
        "Region", back_populates="documents"
    )
    charts: Mapped[list["Chart"]] = relationship(
        "Chart", back_populates="document", cascade="all, delete-orphan"
    )
    images: Mapped[list["Image"]] = relationship(
        "Image", back_populates="document", cascade="all, delete-orphan"
    )
    entities: Mapped[list["Entity"]] = relationship(
        "Entity", back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )
    facets: Mapped[list["DocumentFacet"]] = relationship(
        "DocumentFacet", back_populates="document", cascade="all, delete-orphan"
    )
    keywords: Mapped[list["Keyword"]] = relationship(
        "Keyword", back_populates="document", cascade="all, delete-orphan"
    )
    client: Mapped["Client | None"] = relationship("Client", back_populates="documents")

    @property
    def client_name(self) -> str | None:
        from sqlalchemy import inspect as sa_inspect

        state = sa_inspect(self)
        if "client" not in state.unloaded and self.client is not None:
            return self.client.name
        return None

    __table_args__ = (Index("idx_document_status", "status"),)

    @property
    def period(self) -> str | None:
        """Alias for report_period for convenience."""
        return self.report_period


# =============================================================================
# Section Model (Hierarchical Structure)
# =============================================================================


class Section(Base):
    """
    Logical section/grouping within a document.

    Sections provide hierarchical organization (e.g., "Performance Overview",
    "Market Analysis", "Recommendations") and can be nested.
    """

    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sections.id", ondelete="CASCADE"), nullable=True
    )

    # Section info
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    order: Mapped[int] = mapped_column(Integer, default=0)

    # Slide range this section covers
    start_slide: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_slide: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # LLM-enhanced summary
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_points: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="sections")
    parent: Mapped["Section | None"] = relationship(
        "Section", remote_side=[id], back_populates="children"
    )
    children: Mapped[list["Section"]] = relationship(
        "Section", back_populates="parent", cascade="all, delete-orphan"
    )
    slides: Mapped[list["Slide"]] = relationship("Slide", back_populates="section")

    __table_args__ = (
        Index("idx_section_document", "document_id"),
        Index("idx_section_parent", "parent_id"),
    )


# =============================================================================
# Slide Model
# =============================================================================


class Slide(Base):
    """
    Individual slide/page within a document.

    Contains both raw extraction data and LLM-enhanced analysis.
    """

    __tablename__ = "slides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("sections.id", ondelete="SET NULL"), nullable=True
    )

    # Slide position
    slide_number: Mapped[int] = mapped_column(Integer, nullable=False)
    google_slide_id: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Raw content
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    speaker_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # LLM-enhanced fields
    slide_type: Mapped[str] = mapped_column(String(50), default=SlideType.UNKNOWN.value)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    key_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    insights: Mapped[list | None] = mapped_column(JSON, nullable=True)
    action_items: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Content flags
    has_images: Mapped[bool] = mapped_column(Boolean, default=False)
    has_charts: Mapped[bool] = mapped_column(Boolean, default=False)
    has_tables: Mapped[bool] = mapped_column(Boolean, default=False)
    image_count: Mapped[int] = mapped_column(Integer, default=0)

    # LLM confidence
    classification_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )

    # Byte positions in original content
    byte_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    byte_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="slides")
    section: Mapped["Section | None"] = relationship("Section", back_populates="slides")
    metrics: Mapped[list["Metric"]] = relationship("Metric", back_populates="slide")
    charts: Mapped[list["Chart"]] = relationship("Chart", back_populates="slide")
    images: Mapped[list["Image"]] = relationship("Image", back_populates="slide")
    slide_entities: Mapped[list["SlideEntity"]] = relationship(
        "SlideEntity", back_populates="slide", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_id", "slide_number", name="uq_slide_number"),
        Index("idx_slide_document", "document_id"),
        Index("idx_slide_type", "slide_type"),
        Index("idx_slide_section", "section_id"),
    )


# =============================================================================
# Metric Model
# =============================================================================


class Period(Base):
    """Normalized reporting period."""

    __tablename__ = "periods"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    period_label: Mapped[str] = mapped_column(String(50), nullable=False)
    period_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    period_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fiscal_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_date: Mapped[str | None] = mapped_column(String(20), nullable=True)
    end_date: Mapped[str | None] = mapped_column(String(20), nullable=True)

    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", back_populates="period"
    )

    __table_args__ = (
        UniqueConstraint(
            "period_label",
            "start_date",
            "end_date",
            name="uq_period_label_range",
        ),
        Index("idx_period_label", "period_label"),
        Index("idx_period_fiscal", "fiscal_year"),
        Index("idx_period_type_number", "period_type", "period_number"),
    )


class MetricCatalog(Base):
    """Canonical metric catalog with normalized definitions."""

    __tablename__ = "metric_catalog"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    slug: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(50), nullable=True)
    default_unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    formula: Mapped[str | None] = mapped_column(Text, nullable=True)
    applicability_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    aliases: Mapped[list["MetricAlias"]] = relationship(
        "MetricAlias", back_populates="metric", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["Metric"]] = relationship(
        "Metric", back_populates="metric_catalog"
    )


class MetricAlias(Base):
    """Alias/patterns for matching metric names."""

    __tablename__ = "metric_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("metric_catalog.id", ondelete="CASCADE"), nullable=False
    )
    alias: Mapped[str] = mapped_column(String(200), nullable=False)
    pattern: Mapped[str | None] = mapped_column(String(500), nullable=True)
    priority: Mapped[int] = mapped_column(Integer, default=0)
    unit_override: Mapped[str | None] = mapped_column(String(50), nullable=True)

    metric: Mapped["MetricCatalog"] = relationship("MetricCatalog", back_populates="aliases")


class Metric(Base):
    """
    Extracted and normalized business metric.

    Contains both raw extracted value and LLM-enhanced interpretation.
    """

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    slide_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("slides.id", ondelete="SET NULL"), nullable=True
    )

    # Raw extraction
    raw_value: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_metric_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # percentage, currency, rate
    metric_catalog_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("metric_catalog.id", ondelete="SET NULL"), nullable=True
    )

    # LLM-enhanced normalization
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    normalized_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit: Mapped[str | None] = mapped_column(String(50), nullable=True)
    category: Mapped[str] = mapped_column(
        String(50), default=MetricCategory.OTHER.value
    )

    # Confidence
    extraction_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Period/brand/baseline context
    period_label: Mapped[str | None] = mapped_column(String(50), nullable=True)
    period_start: Mapped[str | None] = mapped_column(String(20), nullable=True)
    period_end: Mapped[str | None] = mapped_column(String(20), nullable=True)
    brand: Mapped[str | None] = mapped_column(String(200), nullable=True)
    baseline_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    baseline_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    period_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("periods.id", ondelete="SET NULL"), nullable=True
    )
    region_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("regions.id", ondelete="SET NULL"), nullable=True
    )
    country: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="metrics")
    slide: Mapped["Slide | None"] = relationship("Slide", back_populates="metrics")
    period: Mapped["Period | None"] = relationship("Period", back_populates="metrics")
    metric_catalog: Mapped["MetricCatalog | None"] = relationship(
        "MetricCatalog", back_populates="metrics"
    )
    region: Mapped["Region | None"] = relationship("Region", back_populates="metrics")

    __table_args__ = (
        Index("idx_metric_document", "document_id"),
        Index("idx_metric_category", "category"),
        Index("idx_metric_name", "name"),
        Index("idx_metric_region", "region_id"),
    )


# =============================================================================
# Metric Fact Embeddings
# =============================================================================


class MetricFactEmbedding(Base):
    """Stored embeddings for metric facts (one per metric row)."""

    __tablename__ = "metric_fact_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    metric_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("metrics.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    text_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    metric: Mapped["Metric"] = relationship("Metric")

    __table_args__ = (Index("idx_metric_fact_embedding_metric", "metric_id"),)


# =============================================================================
# Chart Model
# =============================================================================


class Chart(Base):
    """
    Reconstructed chart/visualization data.

    Contains both detected patterns and LLM-reconstructed structured data.
    """

    __tablename__ = "charts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    slide_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("slides.id", ondelete="SET NULL"), nullable=True
    )

    # Detection info
    chart_type: Mapped[str] = mapped_column(String(50), default=ChartType.UNKNOWN.value)
    detection_confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # Raw elements
    raw_elements: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # LLM-reconstructed data
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    x_axis_label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    y_axis_label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    data_series: Mapped[list | None] = mapped_column(
        JSON, nullable=True
    )  # [{label, values}]
    data_table: Mapped[list | None] = mapped_column(
        JSON, nullable=True
    )  # [[col1, col2, ...], ...]

    # Insights
    insights: Mapped[list | None] = mapped_column(JSON, nullable=True)
    trends_identified: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="charts")
    slide: Mapped["Slide | None"] = relationship("Slide", back_populates="charts")

    __table_args__ = (
        Index("idx_chart_document", "document_id"),
        Index("idx_chart_type", "chart_type"),
    )


# =============================================================================
# Image Model
# =============================================================================


class Image(Base):
    """
    Extracted image with LLM analysis.
    """

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    slide_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("slides.id", ondelete="SET NULL"), nullable=True
    )

    # File info
    filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Position
    image_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # LLM analysis
    image_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # chart, photo, logo, diagram
    contains_text: Mapped[bool] = mapped_column(Boolean, default=False)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # If chart image
    chart_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="images")
    slide: Mapped["Slide | None"] = relationship("Slide", back_populates="images")

    __table_args__ = (Index("idx_image_document", "document_id"),)


# =============================================================================
# Entity Model (Named Entity Recognition)
# =============================================================================


class Entity(Base):
    """
    Named entity extracted from the document.

    Supports building a knowledge graph of companies, people, products, etc.
    """

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    # Entity info
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    normalized_name: Mapped[str | None] = mapped_column(
        String(500), nullable=True
    )  # Canonical form

    # Additional context
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Occurrence count
    mention_count: Mapped[int] = mapped_column(Integer, default=1)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="entities")
    slide_mentions: Mapped[list["SlideEntity"]] = relationship(
        "SlideEntity", back_populates="entity", cascade="all, delete-orphan"
    )
    source_relations: Mapped[list["EntityRelation"]] = relationship(
        "EntityRelation",
        foreign_keys="EntityRelation.source_entity_id",
        back_populates="source_entity",
        cascade="all, delete-orphan",
    )
    target_relations: Mapped[list["EntityRelation"]] = relationship(
        "EntityRelation",
        foreign_keys="EntityRelation.target_entity_id",
        back_populates="target_entity",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "name", "entity_type", name="uq_entity"),
        Index("idx_entity_document", "document_id"),
        Index("idx_entity_type", "entity_type"),
        Index("idx_entity_name", "name"),
    )


class SlideEntity(Base):
    """
    Association between slides and entities (many-to-many with context).
    """

    __tablename__ = "slide_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    slide_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("slides.id", ondelete="CASCADE"), nullable=False
    )
    entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )

    # Context of mention
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    role: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # subject, comparison, etc.

    # Relationships
    slide: Mapped["Slide"] = relationship("Slide", back_populates="slide_entities")
    entity: Mapped["Entity"] = relationship("Entity", back_populates="slide_mentions")

    __table_args__ = (
        UniqueConstraint("slide_id", "entity_id", name="uq_slide_entity"),
        Index("idx_slide_entity_slide", "slide_id"),
        Index("idx_slide_entity_entity", "entity_id"),
    )

    @property
    def mention_context(self) -> str | None:
        """Alias for context for convenience."""
        return self.context


class EntityRelation(Base):
    """
    Relationships between entities (knowledge graph edges).
    """

    __tablename__ = "entity_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )

    # Relation type
    relation_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )  # owns, competes_with, markets_in, etc.
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    source_entity: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[source_entity_id], back_populates="source_relations"
    )
    target_entity: Mapped["Entity"] = relationship(
        "Entity", foreign_keys=[target_entity_id], back_populates="target_relations"
    )

    __table_args__ = (
        Index("idx_relation_source", "source_entity_id"),
        Index("idx_relation_target", "target_entity_id"),
        Index("idx_relation_type", "relation_type"),
    )


# =============================================================================
# Chunk Model (RAG)
# =============================================================================


class Chunk(Base):
    """
    Text chunk for RAG/retrieval purposes.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    # Chunk content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Position info
    byte_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    byte_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    # Embedding (stored as JSON array for simplicity)
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # LLM-enhanced metadata
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    topics: Mapped[list | None] = mapped_column(JSON, nullable=True)
    importance_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Slide coverage
    start_slide: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_slide: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_document", "document_id"),
        Index("idx_chunk_index", "chunk_index"),
    )


# =============================================================================
# Facet Models (For Filtering/Search)
# =============================================================================


class Facet(Base):
    """
    Global facet definition for filtering.

    Facets are dimensions like "Market", "Campaign", "Time Period" that
    can be used to filter and navigate documents.
    """

    __tablename__ = "facets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    facet_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Values for this facet
    values: Mapped[list["FacetValue"]] = relationship(
        "FacetValue", back_populates="facet", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_facet_type", "facet_type"),)


class FacetValue(Base):
    """
    Specific value within a facet.
    """

    __tablename__ = "facet_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    facet_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("facets.id", ondelete="CASCADE"), nullable=False
    )

    value: Mapped[str] = mapped_column(String(500), nullable=False)
    normalized_value: Mapped[str | None] = mapped_column(String(500), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    facet: Mapped["Facet"] = relationship("Facet", back_populates="values")
    document_facets: Mapped[list["DocumentFacet"]] = relationship(
        "DocumentFacet", back_populates="facet_value", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("facet_id", "value", name="uq_facet_value"),
        Index("idx_facet_value_facet", "facet_id"),
    )


class DocumentFacet(Base):
    """
    Association between documents and facet values.

    Enables faceted search/filtering of documents.
    """

    __tablename__ = "document_facets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    facet_value_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("facet_values.id", ondelete="CASCADE"), nullable=False
    )

    # Optional: slide-level faceting
    slide_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("slides.id", ondelete="SET NULL"), nullable=True
    )

    # Confidence if auto-detected
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="facets")
    facet_value: Mapped["FacetValue"] = relationship(
        "FacetValue", back_populates="document_facets"
    )

    __table_args__ = (
        Index("idx_doc_facet_document", "document_id"),
        Index("idx_doc_facet_value", "facet_value_id"),
    )


# =============================================================================
# Keyword Model
# =============================================================================


class Keyword(Base):
    """
    Extracted keyword/topic from document.
    """

    __tablename__ = "keywords"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )

    keyword: Mapped[str] = mapped_column(String(200), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    category: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # business_term, topic, etc.

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="keywords")

    __table_args__ = (
        Index("idx_keyword_document", "document_id"),
        Index("idx_keyword_keyword", "keyword"),
    )
