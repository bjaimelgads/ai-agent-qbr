"""Pydantic models for ai-agent-qbr tools."""

from __future__ import annotations

from pydantic import BaseModel


class Query(BaseModel):
    """User query."""

    question: str
    comparison_intent: bool | None = None


class SearchResult(BaseModel):
    """Single search hit."""

    title: str
    snippet: str
    chunk_id: int | None = None
    document_id: int | None = None
    score: float | None = None
    slide_range: str | None = None


class SearchResults(BaseModel):
    """List of search results."""

    results: list[SearchResult]


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


class ComparisonIntentArgs(BaseModel):
    """Arguments for comparison intent detection."""

    question: str


class ComparisonIntentResult(BaseModel):
    """Comparison intent detection result."""

    comparison_intent: bool
    reason: str | None = None
