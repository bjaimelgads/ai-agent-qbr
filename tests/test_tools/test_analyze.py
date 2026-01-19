"""Unit tests for analyze tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ai_agent_qbr.models import FinalAnswer, SearchResult, SearchResults
from ai_agent_qbr.tools.analyze import analyze_results

if TYPE_CHECKING:
    from conftest import DummyToolContext


@pytest.mark.asyncio
async def test_analyze_results_summarizes_hits(dummy_ctx: "DummyToolContext") -> None:
    """Verify analyze_results produces a summary from search results."""
    hits = SearchResults(
        results=[
            SearchResult(title="PenguiFlow Basics", snippet="An async-first orchestration library"),
            SearchResult(title="Getting Started", snippet="Install with pip install penguiflow"),
        ]
    )

    result = await analyze_results(hits, dummy_ctx)

    assert isinstance(result, FinalAnswer)
    assert result.text, "Expected non-empty answer text"


@pytest.mark.asyncio
async def test_analyze_results_handles_single_result(dummy_ctx: "DummyToolContext") -> None:
    """Verify analyze works with a single search result."""
    hits = SearchResults(
        results=[SearchResult(title="Single Result", snippet="Only one item")]
    )

    result = await analyze_results(hits, dummy_ctx)

    assert isinstance(result, FinalAnswer)
    assert result.text


@pytest.mark.asyncio
async def test_analyze_results_handles_empty_results(dummy_ctx: "DummyToolContext") -> None:
    """Verify analyze handles empty search results gracefully."""
    hits = SearchResults(results=[])

    result = await analyze_results(hits, dummy_ctx)

    assert isinstance(result, FinalAnswer)
    # Should still produce some output even with no results


