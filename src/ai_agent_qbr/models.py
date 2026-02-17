"""Pydantic models for ai-agent-qbr tools."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class Query(BaseModel):
    """User query."""

    question: str
    comparison_intent: bool | None = None


class SearchResult(BaseModel):
    """Single search hit."""

    title: str
    document_title: str | None = None
    slide_title: str | None = None
    snippet: str
    chunk_id: int | None = None
    document_id: int | None = None
    score: float | None = None
    slide_range: str | None = None
    document_url: str | None = None
    slide_url: str | None = None
    source_url: str | None = None


class SearchResults(BaseModel):
    """List of search results."""

    results: list[SearchResult]
    needs_clarification: bool = False
    clarification_question: str | None = None
    suggested_filters: list[str] = Field(default_factory=list)
    candidate_document_count: int | None = None


class FinalAnswer(BaseModel):
    """Final agent answer."""

    text: str


class AgentCapabilitiesArgs(BaseModel):
    """Arguments for agent_capabilities tool."""

    include_examples: bool = True


class AgentCapabilitiesResult(BaseModel):
    """Capabilities response for users."""

    capabilities_text: str
    sample_queries: list[str]


class MetadataCatalogArgs(BaseModel):
    """Arguments for metadata_catalog tool."""

    question: str
    limit: int = Field(default=20, ge=1, le=200)


class MetadataCatalogResult(BaseModel):
    """Structured metadata catalog response."""

    summary_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    suggested_queries: list[str] = Field(default_factory=list)


class ComparisonIntentArgs(BaseModel):
    """Arguments for comparison intent detection."""

    question: str


class ComparisonIntentResult(BaseModel):
    """Comparison intent detection result."""

    comparison_intent: bool
    reason: str | None = None


class RegionFilterVerificationArgs(BaseModel):
    """Arguments for region filter verification."""

    question: str


class RegionFilterVerificationResult(BaseModel):
    """Region filter verification result."""

    region_focus: str | None = None
    needs_clarification: bool = False
    reason: str | None = None
    confidence: float | None = None


class MetricQueryArgs(BaseModel):
    """Arguments for metric query tool."""

    question: str
    debug: bool = False
    intent: ResolvedMetricIntent | None = None


class PeriodSpec(BaseModel):
    """Normalized period specification."""

    type: str | None = None
    value: str | None = None
    start: date | None = None
    end: date | None = None


class ResolvedMetricIntent(BaseModel):
    """Structured intent resolved from a metric query."""

    metric_ids: list[str] = Field(default_factory=list)
    client: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=list)
    period: list[PeriodSpec] = Field(default_factory=list)
    aggregation: str | None = None
    grouping: str | None = None
    limit: int | None = None


class FieldConfidence(BaseModel):
    """Confidence scores for resolved intent fields."""

    metric: float = 0.0
    client: float = 0.0
    region: float = 0.0
    period: float = 0.0


class ResolveMetricIntentArgs(BaseModel):
    """Arguments for resolve_metric_intent tool."""

    question: str
    proposed_entities: CandidateSet | None = None


class ResolveMetricIntentResult(BaseModel):
    """Resolved metric intent with confidence metadata."""

    intent: ResolvedMetricIntent
    confidence: FieldConfidence


class CandidateSet(BaseModel):
    """Candidate values proposed by the planner."""

    metric_ids: list[str] = Field(default_factory=list)
    clients: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)


class RefineMetricIntentArgs(BaseModel):
    """Arguments for refine_metric_intent tool."""

    question: str
    intent: ResolvedMetricIntent
    candidates: CandidateSet


class RefineMetricIntentResult(BaseModel):
    """Validated metric intent after applying candidate refinements."""

    intent: ResolvedMetricIntent
    assumptions: list[str] = Field(default_factory=list)
    unresolved_fields: list[str] = Field(default_factory=list)
