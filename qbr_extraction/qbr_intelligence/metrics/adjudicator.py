"""Optional LLM adjudicator for low-confidence metric links."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

from qbr_intelligence.metrics.models import MetricExtraction


@dataclass(frozen=True)
class AdjudicationRequest:
    deck_hash: str
    slide_index: int
    metric_id: str
    metric_name: str
    label_text: str
    context_snippet: str
    candidates: list[dict]

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "deck_hash": self.deck_hash,
                "slide_index": self.slide_index,
                "metric_id": self.metric_id,
                "label_text": self.label_text,
                "candidates": self.candidates,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AdjudicationResult:
    chosen_index: int
    normalized_value: float | None
    unit: str
    reasoning_short: str


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


class LLMAdjudicator:
    """Minimal LLM adjudicator with file cache."""

    def __init__(
        self,
        *,
        call_llm: Callable[[str], str] | None,
        cache_dir: str | Path,
        enabled: bool = True,
    ) -> None:
        self._call_llm = call_llm
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._enabled = enabled and call_llm is not None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResult | None:
        if not self._enabled:
            return None
        cache_key = request.cache_key()
        cache_path = self._cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return AdjudicationResult(
                    chosen_index=int(payload.get("chosen_index", -1)),
                    normalized_value=payload.get("normalized_value"),
                    unit=payload.get("unit") or "unknown",
                    reasoning_short=payload.get("reasoning_short", "cached"),
                )
            except Exception:
                pass

        prompt = _build_prompt(request)
        raw = self._call_llm(prompt)
        payload = _extract_json(raw or "")
        if not payload:
            return None
        result = AdjudicationResult(
            chosen_index=int(payload.get("chosen_index", -1)),
            normalized_value=payload.get("normalized_value"),
            unit=payload.get("unit") or "unknown",
            reasoning_short=payload.get("reasoning_short", ""),
        )
        cache_path.write_text(
            json.dumps(
                {
                    "chosen_index": result.chosen_index,
                    "normalized_value": result.normalized_value,
                    "unit": result.unit,
                    "reasoning_short": result.reasoning_short,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return result


def _build_prompt(request: AdjudicationRequest) -> str:
    return (
        "You are adjudicating metric extraction candidates. "
        "Return strict JSON only with keys: chosen_index, normalized_value, unit, reasoning_short. "
        "If none are correct, chosen_index should be -1.\n"
        f"Metric: {request.metric_name} ({request.metric_id})\n"
        f"Label: {request.label_text}\n"
        f"Context: {request.context_snippet}\n"
        f"Candidates JSON: {json.dumps(request.candidates, indent=2)}\n"
    )
