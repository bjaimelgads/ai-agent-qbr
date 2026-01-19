"""Pydantic models for ai-agent-qbr tools."""

from __future__ import annotations

from pydantic import BaseModel


class Query(BaseModel):
    """User query."""

    question: str


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
