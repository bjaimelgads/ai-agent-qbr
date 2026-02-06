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
    client: str | None = None
    region: str | None = None
    period: PeriodSpec | None = None
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


class AnswerCitation(BaseModel):
    """Citation for metric answers."""

    document_id: int
    document_name: str | None = None
    document_url: str | None = None
    slide_id: int | None = None
    slide_number: int | None = None
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
