"""Comparison intent detection tool."""

from __future__ import annotations

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

from ai_agent_qbr.models import ComparisonIntentArgs, ComparisonIntentResult

_KEYWORD_HINTS = (
    "compare",
    "comparison",
    "versus",
    "vs ",
    "difference",
    "delta",
    "between",
    "across",
    "relative to",
    "year over year",
    "yoy",
    "quarter over quarter",
    "qoq",
    "month over month",
    "mom",
    "prior period",
    "trend across",
    "vs.",
)

_DECK_HINTS = (
    "deck",
    "presentation",
    "pptx",
    ".pptx",
    "slides from",
)


@tool(
    desc=(
        "Determine whether the question requires comparing information across multiple decks "
        "or presentations. Call this when you suspect the answer needs cross-deck comparison, "
        "so retrieval can search more than one document."
    ),
    side_effects="read",
    tags=["planner"],
)
async def detect_comparison_intent(
    args: ComparisonIntentArgs,
    _ctx: ToolContext,
) -> ComparisonIntentResult:
    question = args.question.strip()
    lowered = question.lower()
    reason = None

    for hint in _KEYWORD_HINTS:
        if hint in lowered:
            reason = f"keyword:{hint.strip()}"
            break

    deck_hits = sum(1 for hint in _DECK_HINTS if hint in lowered)
    if deck_hits >= 2 and reason is None:
        reason = "mentions multiple deck indicators"
    elif deck_hits >= 1 and ("compare" in lowered or "versus" in lowered or "vs" in lowered):
        reason = "deck reference with comparison cue"

    comparison_intent = reason is not None
    return ComparisonIntentResult(comparison_intent=comparison_intent, reason=reason)
