"""Optional LLM adjudicator for low-confidence metric links."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import os
import re
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
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        literal = ast.literal_eval(text)
        if isinstance(literal, (list, tuple)) and literal:
            if isinstance(literal[0], str):
                text = literal[0]
        elif isinstance(literal, str):
            text = literal
    except Exception:
        pass
    if not isinstance(text, str):
        return None
    fence_matches = re.findall(r"```(?:json)?\\s*(\\{.*?\\})\\s*```", text, re.DOTALL)
    if fence_matches:
        text = fence_matches[-1]
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
        self.total_requests = 0
        self.cache_hits = 0
        self.llm_calls = 0
        self.parse_failures = 0
        self.empty_responses = 0
        self._dump_failures = (os.getenv("METRIC_ADJUDICATOR_DUMP") or "").lower() in {
            "1",
            "true",
            "yes",
        }
        self._dump_limit = int(os.getenv("METRIC_ADJUDICATOR_DUMP_LIMIT", "3") or "3")
        self._dump_count = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def adjudicate(self, request: AdjudicationRequest) -> AdjudicationResult | None:
        if not self._enabled:
            return None
        self.total_requests += 1
        cache_key = request.cache_key()
        cache_path = self._cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                self.cache_hits += 1
                return AdjudicationResult(
                    chosen_index=int(payload.get("chosen_index", -1)),
                    normalized_value=payload.get("normalized_value"),
                    unit=payload.get("unit") or "unknown",
                    reasoning_short=payload.get("reasoning_short", "cached"),
                )
            except Exception:
                pass

        prompt = _build_prompt(request)
        self.llm_calls += 1
        raw = self._call_llm(prompt)
        if not raw or not str(raw).strip():
            self.empty_responses += 1
            self.parse_failures += 1
            self._maybe_dump_failure(request, raw)
            return None
        payload = _extract_json(raw or "")
        if not payload:
            self.parse_failures += 1
            self._maybe_dump_failure(request, raw)
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

    def _maybe_dump_failure(self, request: AdjudicationRequest, raw: str | None) -> None:
        if not self._dump_failures:
            return
        if self._dump_count >= self._dump_limit:
            return
        self._dump_count += 1
        record = {
            "metric_id": request.metric_id,
            "metric_name": request.metric_name,
            "label_text": request.label_text,
            "slide_index": request.slide_index,
            "prompt": _shorten(_build_prompt(request), 4000),
            "raw_response": _shorten(raw or "", 4000),
        }
        target = self._cache_dir / f"adjudicator_failure_{self._dump_count}.json"
        _dump_record(target, record)


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


def _shorten(text: str, limit: int = 4000) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit]


def _dump_record(path: Path, record: dict) -> None:
    path.write_text(
        json.dumps(record, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
