"""Region filtering helpers for QBR context."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_EMEA_CONTENT_PATTERNS = (
    r"\bEMEA\b",
    r"\bEU5\b",
    r"\bEU-?5\b",
    r"\bEU\b",
    r"\bEurope(?:an)?\b",
    r"\bUK\b",
    r"\bU\.K\.\b",
    r"\bGermany\b",
    r"\bFrance\b",
    r"\bItaly\b",
    r"\bSpain\b",
    r"\bTurkey\b",
    r"\bPoland\b",
    r"\bSweden\b",
    r"\bGreece\b",
    r"\bDE\b",
    r"\bFR\b",
    r"\bIT\b",
    r"\bES\b",
    r"\bTR\b",
    r"\bPL\b",
    r"\bSE\b",
    r"\bGR\b",
)

_US_CONTENT_CASE_SENSITIVE = (
    r"\bUS\b",
    r"\bUSA\b",
)

_US_CONTENT_CASE_INSENSITIVE = (
    r"\bU\.S\.\b",
    r"\bU\.S\.A\.\b",
    r"\bUnited States\b",
    r"\bDomestic\b",
    r"\bNorth America\b",
)


def filter_items_by_region(items: Iterable[Any], region_focus: str | None) -> list[Any]:
    """Filter items whose content mentions a different region than requested."""
    if not region_focus:
        return list(items)

    filtered: list[Any] = []
    for item in items:
        content = _extract_content(item)
        if region_focus == "us":
            if _mentions_emea(content):
                continue
        elif region_focus == "emea":
            if _mentions_us(content):
                continue
        filtered.append(item)
    return filtered


def build_context_from_items(items: Iterable[Any]) -> str:
    """Rebuild context text from filtered items."""
    parts = [content for item in items if (content := _extract_content(item))]
    return "\n\n".join(parts)


def _extract_content(item: Any) -> str:
    if item is None:
        return ""
    chunk = getattr(item, "chunk", None)
    if chunk is not None:
        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content
    if isinstance(item, dict):
        content = item.get("content")
        if isinstance(content, str):
            return content
        chunk = item.get("chunk")
        if isinstance(chunk, dict):
            chunk_content = chunk.get("content")
            if isinstance(chunk_content, str):
                return chunk_content
    return ""


def _mentions_emea(content: str) -> bool:
    return _matches_any(content, _EMEA_CONTENT_PATTERNS, re.IGNORECASE)


def _mentions_us(content: str) -> bool:
    if _matches_any(content, _US_CONTENT_CASE_SENSITIVE, 0):
        return True
    return _matches_any(content, _US_CONTENT_CASE_INSENSITIVE, re.IGNORECASE)


def _matches_any(text: str, patterns: Iterable[str], flags: int) -> bool:
    return any(re.search(pattern, text, flags) for pattern in patterns)
