"""Helpers to normalize planner-provided question arguments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def extract_canonical_query(question: str) -> str:
    """Extract the most likely user query from a planner wrapper payload."""
    text = (question or "").strip()
    if not text:
        return text
    if not text.startswith("{"):
        return text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text

    queue: list[dict[str, Any]] = [payload]
    while queue:
        item = queue.pop(0)
        for key in ("query", "question"):
            value = item.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate
        for value in item.values():
            if isinstance(value, dict):
                queue.append(value)
    return text


def normalize_question_arg(raw_question: str, tool_context: Mapping[str, Any] | None) -> str:
    """Prefer explicit user query from payload; otherwise fallback to original_query."""
    canonical = extract_canonical_query(raw_question)
    if canonical.lstrip().startswith("{"):
        fallback = str((tool_context or {}).get("original_query") or "").strip()
        if fallback:
            return fallback
    return canonical
