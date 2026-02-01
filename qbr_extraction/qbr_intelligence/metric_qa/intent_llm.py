"""LLM-backed intent parsing for metric QA."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

def build_intent_llm(
    *,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Callable[[str, dict], dict | None]:
    from qbr_intelligence.llm.modules import create_lm

    lm = create_lm(model=model, max_tokens=max_tokens, temperature=temperature)

    def _call(query: str, payload: dict) -> dict | None:
        prompt = _build_prompt(query, payload)
        text = _call_llm_text(lm, prompt)
        if not text:
            return None
        return _extract_json(text)

    return _call


def _build_prompt(query: str, payload: dict) -> str:
    metrics = payload.get("metrics", [])
    regions = payload.get("regions", [])
    aggregations = payload.get("aggregations", [])
    period_examples = payload.get("period_examples", [])

    return (
        "You are a JSON-only intent parser for metric questions.\n"
        "Return a single JSON object with this schema:\n"
        "{\n"
        '  "metric_ids": [string],\n'
        '  "client": string|null,\n'
        '  "region": string|null,\n'
        '  "period": {"type": "quarter|half|year|range|relative"|null, "value": string|null, "start": string|null, "end": string|null} | null,\n'
        '  "aggregation": "latest|trend|compare|average|sum|min|max"|null,\n'
        '  "grouping": "by_period|by_region|by_client"|null,\n'
        '  "limit": number|null\n'
        "}\n"
        "Rules:\n"
        "- Use only provided metric_ids.\n"
        "- If unsure, use null or empty list.\n"
        "- Dates must be ISO format (YYYY-MM-DD) if provided.\n"
        "- Output JSON only, no prose.\n"
        f"\nKnown metrics (id + aliases): {json.dumps(metrics, separators=(',', ':'))}\n"
        f"Known regions: {json.dumps(regions, separators=(',', ':'))}\n"
        f"Aggregations: {json.dumps(aggregations, separators=(',', ':'))}\n"
        f"Period examples: {json.dumps(period_examples, separators=(',', ':'))}\n"
        f"\nUser query: {query}\n"
        "JSON:"
    )


def _extract_json(text: str) -> dict | None:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _call_llm_text(lm: Any, prompt: str) -> str:
    def _extract_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, (list, tuple)) and response:
            if isinstance(response[0], str):
                return response[0]
        if isinstance(response, dict):
            choices = response.get("choices")
            if choices:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message") or {}
                    if isinstance(message, dict) and message.get("content"):
                        return message["content"]
                    if choice.get("text"):
                        return choice["text"]
            if response.get("content"):
                return response["content"]
        choices = getattr(response, "choices", None)
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
            text = getattr(choice, "text", None)
            if isinstance(text, str):
                return text
        for attr in ("text", "completion", "content", "output_text"):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        return ""

    for method_name in ("request", "complete", "__call__"):
        method = getattr(lm, method_name, None)
        if not callable(method):
            continue
        try:
            response = method(prompt)
        except Exception:
            continue
        text = _extract_text(response)
        if text:
            return text
        try:
            return str(response)
        except Exception:
            continue
    return ""
