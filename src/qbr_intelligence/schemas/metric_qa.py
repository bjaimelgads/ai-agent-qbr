"""Schemas for structured metric QA."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class PeriodSpec(BaseModel):
    """Normalized period specification."""

    type: Literal["quarter", "half", "year", "range", "relative"] | None = None
    value: str | None = None
    start: date | None = None
    end: date | None = None


class QueryIntent(BaseModel):
    """Structured intent extracted from a metric query."""

    metric_ids: list[str] = Field(default_factory=list)
    client: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=list)
    period: list[PeriodSpec] = Field(default_factory=list)
    aggregation: Literal[
        "latest",
        "all",
        "average",
        "trend",
        "compare",
        "sum",
        "min",
        "max",
    ] | None = None
    grouping: Literal["by_period", "by_region", "by_client"] | None = None
    limit: int | None = None
    clarifications_needed: list[str] = Field(default_factory=list)


class MetricRow(BaseModel):
    metric: str | None = None
    value: float | None = None
    unit: str | None = None
    period: str | None = None
    client: str | None = None
    region: str | None = None
    llm_context_label: str | None = None
    semantic_score: float | None = None


class AnswerCitation(BaseModel):
    """Citation for metric answers."""

    document_id: int
    document_name: str | None = None
    document_url: str | None = None
    slide_id: int | None = None
    slide_number: int | None = None
    slide_title: str | None = None
    slide_google_id: str | None = None
    slide_url: str | None = None
    snippet: str | None = None


class MetricAnswer(BaseModel):
    """Structured response for metric QA."""

    summary: str | None = None
    summary_text: str
    data: list[MetricRow] | None = None
    table_data: list[dict] | None = None
    chart_spec: dict | None = None
    citations: list[AnswerCitation] = Field(default_factory=list)
    confidence: float | None = None
    assumptions: list[str] = Field(default_factory=list)
    followups: list[str] = Field(default_factory=list)
    debug: dict | None = None


class MetricEvidenceIntent(BaseModel):
    """Intent payload forwarded to planner for evidence-first workflows."""

    metric_ids: list[str] = Field(default_factory=list)
    client: list[str] = Field(default_factory=list)
    region: list[str] = Field(default_factory=list)
    period: list[str] = Field(default_factory=list)


class MetricEvidenceSource(BaseModel):
    """Document/slide source descriptor."""

    document_id: int | None = None
    document_name: str | None = None
    document_url: str | None = None
    slide_id: int | None = None
    slide_number: int | None = None
    slide_title: str | None = None
    slide_url: str | None = None


class MetricEvidenceMetric(BaseModel):
    """Metric value evidence row."""

    metric_id: str | None = None
    metric_name: str | None = None
    value: float | None = None
    unit: str | None = None
    period: str | None = None
    client: str | None = None
    region: str | None = None
    context_label: str | None = None
    confidence: float | None = None
    source: MetricEvidenceSource = Field(default_factory=MetricEvidenceSource)


class MetricEvidenceChunk(BaseModel):
    """Retrieved chunk evidence row."""

    chunk_id: int | None = None
    document_id: int | None = None
    document_name: str | None = None
    document_url: str | None = None
    start_slide: int | None = None
    end_slide: int | None = None
    slide_title: str | None = None
    slide_url: str | None = None
    score: float | None = None
    content: str | None = None
    matched_metric_slide: bool = False


class MetricEvidenceLink(BaseModel):
    """Link between a metric evidence row and a retrieved chunk."""

    metric_index: int
    chunk_index: int
    overlap_type: Literal["exact", "range_overlap"]
    link_confidence: float = 1.0


class MetricEvidenceAnswer(BaseModel):
    """Evidence-first planner payload for metric queries."""

    query: str
    intent: MetricEvidenceIntent
    metrics: list[MetricEvidenceMetric] = Field(default_factory=list)
    retrieval_chunks: list[MetricEvidenceChunk] = Field(default_factory=list)
    metric_chunk_links: list[MetricEvidenceLink] = Field(default_factory=list)
    retrieval_debug: dict | None = None
    status: Literal["ok", "empty", "error"] = "ok"
    error: str | None = None
