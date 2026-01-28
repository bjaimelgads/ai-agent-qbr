"""Metric context enrichment strategies (period/brand/baseline)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
import os
import re
from typing import Iterable, Mapping, Sequence


_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_RELATIVE_PERIODS = {
    "yoy": "YoY",
    "year over year": "YoY",
    "last year": "last year",
    "prior year": "prior year",
    "qoq": "QoQ",
    "quarter over quarter": "QoQ",
    "last quarter": "last quarter",
    "previous quarter": "previous quarter",
    "mom": "MoM",
    "month over month": "MoM",
    "last month": "last month",
    "ytd": "YTD",
}

_BASELINE_PATTERNS = [
    r"\b(?:vs\.?|versus|compared to|against)\s+([A-Za-z0-9 ,\-\/]+)",
    r"\b(?:from|since)\s+(last quarter|last year|prior year|previous quarter|previous year)\b",
    r"\b(?:YoY|QoQ|MoM|YTD)\b",
]

_BRAND_PATTERN = re.compile(
    r"\b(?:brand|client|advertiser|partner)\s*[:\-]\s*(?P<brand>[A-Za-z0-9 &+\-\.]{2,})",
    re.IGNORECASE,
)

_HALF_PATTERN = re.compile(
    r"\bH(?P<half>[12])\s*(?:FY)?\s*'?(?P<year>\d{2,4})\b", re.IGNORECASE
)
_HALF_PATTERN_REV = re.compile(
    r"\bFY\s*'?(?P<year>\d{2,4})\s*H(?P<half>[12])\b", re.IGNORECASE
)
_QUARTER_PATTERN = re.compile(
    r"\bQ(?P<quarter>[1-4])\s*(?:FY)?\s*'?(?P<year>\d{2,4})\b", re.IGNORECASE
)
_QUARTER_PATTERN_REV = re.compile(
    r"\b(?P<quarter>[1-4])Q\s*'?(?P<year>\d{2,4})\b", re.IGNORECASE
)
_FY_PATTERN = re.compile(r"\bFY\s*'?(?P<year>\d{2,4})\b", re.IGNORECASE)
_MONTH_RANGE_PATTERN = re.compile(
    r"\b(?P<m1>[A-Za-z]{3,9})\s+(?P<y1>\d{4})\s*(?:-|to|–|—)\s*"
    r"(?P<m2>[A-Za-z]{3,9})\s+(?P<y2>\d{4})\b",
    re.IGNORECASE,
)


def _fiscal_start_month() -> int:
    raw = (os.getenv("FISCAL_YEAR_START_MONTH") or "10").strip()
    try:
        value = int(raw)
    except ValueError:
        return 10
    return value if 1 <= value <= 12 else 10


@dataclass(frozen=True)
class DocumentContext:
    client_name: str | None = None
    report_period: str | None = None


@dataclass(frozen=True)
class MetricContextUpdate:
    period_label: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    brand: str | None = None
    baseline_text: str | None = None
    baseline_type: str | None = None
    source_snippet: str | None = None
    strategy: str | None = None


class MetricContextStrategy:
    """Strategy interface for enriching metric context."""

    name = "base"
    stages: set[str] = {"scanned", "deduped", "refined"}
    requires_llm = False

    def apply(
        self,
        *,
        metrics: Sequence,
        slides: Sequence[dict],
        document_context: DocumentContext,
    ) -> Mapping[str, MetricContextUpdate]:
        raise NotImplementedError


class RuleBasedMetricContextStrategy(MetricContextStrategy):
    name = "rule_based"

    def apply(
        self,
        *,
        metrics: Sequence,
        slides: Sequence[dict],
        document_context: DocumentContext,
    ) -> Mapping[str, MetricContextUpdate]:
        slide_lookup = {s.get("slide_number"): s for s in slides}
        updates: dict[str, MetricContextUpdate] = {}
        for metric in metrics:
            slide = slide_lookup.get(metric.slide_number, {}) if metric.slide_number else {}
            text = _join_text(
                metric.raw_context,
                slide.get("raw_text"),
                slide.get("speaker_notes"),
                document_context.report_period,
            )
            period = _extract_period(text) or (
                _extract_period(document_context.report_period or "")
                if document_context.report_period
                else None
            )
            brand = _extract_brand(text, document_context.client_name)
            baseline_text = metric.baseline_text or _extract_baseline(text)
            baseline_type = metric.baseline_type or infer_baseline_type(baseline_text)
            update = MetricContextUpdate(
                period_label=period.label if period else None,
                period_start=period.start if period else None,
                period_end=period.end if period else None,
                brand=brand,
                baseline_text=baseline_text,
                baseline_type=baseline_type,
                source_snippet=period.source if period and period.source else None,
                strategy=self.name,
            )
            if any(
                value is not None
                for value in (
                    update.period_label,
                    update.brand,
                    update.baseline_text,
                    update.baseline_type,
                )
            ):
                updates[metric.metric_id] = update
        return updates


class LLMMetricsContextStrategy(MetricContextStrategy):
    name = "llm"
    requires_llm = True
    stages: set[str] = {"refined"}

    def __init__(self, extractor):
        self.extractor = extractor

    def apply(
        self,
        *,
        metrics: Sequence,
        slides: Sequence[dict],
        document_context: DocumentContext,
    ) -> Mapping[str, MetricContextUpdate]:
        if not metrics:
            return {}
        slide_lookup = {s.get("slide_number"): s for s in slides}
        metrics_by_slide: dict[int | None, list] = {}
        for metric in metrics:
            metrics_by_slide.setdefault(metric.slide_number, []).append(metric)

        updates: dict[str, MetricContextUpdate] = {}
        for slide_number, slide_metrics in metrics_by_slide.items():
            if slide_number is None:
                continue
            slide = slide_lookup.get(slide_number, {})
            slide_text = slide.get("raw_text", "")
            speaker_notes = slide.get("speaker_notes", "")
            if not slide_text and not speaker_notes:
                continue
            candidates = [
                {
                    "id": m.metric_id,
                    "name": m.name,
                    "raw_value": m.raw_value,
                    "value": m.normalized_value,
                    "unit": m.unit,
                    "context": m.raw_context,
                }
                for m in slide_metrics
            ]
            result = self.extractor(
                slide_text=slide_text,
                speaker_notes=speaker_notes,
                candidates=candidates,
                document_context=_format_document_context(document_context),
            )
            contexts = getattr(result, "contexts", []) or []
            for ctx in contexts:
                metric_id = getattr(ctx, "metric_id", None)
                if not metric_id:
                    continue
                update = MetricContextUpdate(
                    period_label=getattr(ctx, "period_label", None),
                    period_start=getattr(ctx, "period_start", None),
                    period_end=getattr(ctx, "period_end", None),
                    brand=getattr(ctx, "brand", None),
                    baseline_text=getattr(ctx, "baseline_text", None),
                    baseline_type=getattr(ctx, "baseline_type", None),
                    source_snippet=getattr(ctx, "source_snippet", None),
                    strategy=self.name,
                )
                updates[metric_id] = update
        return updates


class HybridMetricContextStrategy(MetricContextStrategy):
    name = "hybrid"
    requires_llm = True
    stages: set[str] = {"refined"}

    def __init__(
        self,
        *,
        rule_strategy: RuleBasedMetricContextStrategy,
        llm_strategy: LLMMetricsContextStrategy | None = None,
    ):
        self.rule_strategy = rule_strategy
        self.llm_strategy = llm_strategy

    def apply(
        self,
        *,
        metrics: Sequence,
        slides: Sequence[dict],
        document_context: DocumentContext,
    ) -> Mapping[str, MetricContextUpdate]:
        rule_updates = dict(
            self.rule_strategy.apply(
                metrics=metrics, slides=slides, document_context=document_context
            )
        )
        if self.llm_strategy is None:
            return rule_updates
        llm_updates = self.llm_strategy.apply(
            metrics=metrics, slides=slides, document_context=document_context
        )
        merged: dict[str, MetricContextUpdate] = {}
        metric_ids = set(rule_updates) | set(llm_updates)
        for metric_id in metric_ids:
            rule_update = rule_updates.get(metric_id)
            llm_update = llm_updates.get(metric_id)
            merged[metric_id] = MetricContextUpdate(
                period_label=(llm_update.period_label if llm_update else None)
                or (rule_update.period_label if rule_update else None),
                period_start=(llm_update.period_start if llm_update else None)
                or (rule_update.period_start if rule_update else None),
                period_end=(llm_update.period_end if llm_update else None)
                or (rule_update.period_end if rule_update else None),
                brand=(llm_update.brand if llm_update else None)
                or (rule_update.brand if rule_update else None),
                baseline_text=(llm_update.baseline_text if llm_update else None)
                or (rule_update.baseline_text if rule_update else None),
                baseline_type=(llm_update.baseline_type if llm_update else None)
                or (rule_update.baseline_type if rule_update else None),
                source_snippet=(llm_update.source_snippet if llm_update else None)
                or (rule_update.source_snippet if rule_update else None),
                strategy=self.name,
            )
        return merged


def apply_context_strategies(
    *,
    metrics: Sequence,
    slides: Sequence[dict],
    document_context: DocumentContext,
    strategies: Iterable[MetricContextStrategy],
    stage: str,
) -> list:
    if not metrics:
        return list(metrics)
    updates_by_strategy: list[Mapping[str, MetricContextUpdate]] = []
    for strategy in strategies:
        if stage not in strategy.stages:
            continue
        updates_by_strategy.append(
            strategy.apply(metrics=metrics, slides=slides, document_context=document_context)
        )

    updated_metrics = []
    for metric in metrics:
        merged = metric
        for updates in updates_by_strategy:
            update = updates.get(metric.metric_id)
            if not update:
                continue
            merged = _apply_update(merged, update)
        updated_metrics.append(merged)
    return updated_metrics


def infer_baseline_type(baseline_text: str | None) -> str | None:
    if not baseline_text:
        return None
    text = baseline_text.lower()
    if "yoy" in text or "year over year" in text or "last year" in text or "prior year" in text:
        return "yoy"
    if "qoq" in text or "quarter over quarter" in text or "last quarter" in text:
        return "qoq"
    if "mom" in text or "month over month" in text or "last month" in text:
        return "mom"
    if "ytd" in text:
        return "ytd"
    if "target" in text:
        return "target"
    if "plan" in text:
        return "plan"
    if "benchmark" in text:
        return "benchmark"
    if "control" in text:
        return "control"
    if "baseline" in text or "vs" in text or "versus" in text or "compared to" in text:
        return "baseline"
    return None


def _apply_update(metric, update: MetricContextUpdate):
    metadata = dict(metric.metadata or {})
    trace = metadata.get("context_trace")
    if trace is None:
        trace = []
    if update.strategy:
        trace.append(
            {
                "strategy": update.strategy,
                "period_label": update.period_label,
                "period_start": update.period_start,
                "period_end": update.period_end,
                "brand": update.brand,
                "baseline_text": update.baseline_text,
                "baseline_type": update.baseline_type,
                "source_snippet": update.source_snippet,
            }
        )
    metadata["context_trace"] = trace
    return replace(
        metric,
        period_label=metric.period_label or update.period_label,
        period_start=metric.period_start or update.period_start,
        period_end=metric.period_end or update.period_end,
        brand=metric.brand or update.brand,
        baseline_text=metric.baseline_text or update.baseline_text,
        baseline_type=metric.baseline_type or update.baseline_type,
        metadata=metadata,
    )


def _extract_brand(text: str, client_name: str | None) -> str | None:
    if not text:
        return client_name
    match = _BRAND_PATTERN.search(text)
    if match:
        return match.group("brand").strip().rstrip(".")
    return client_name


@dataclass(frozen=True)
class _PeriodMatch:
    label: str
    start: str | None
    end: str | None
    source: str | None = None


def _extract_period(text: str) -> _PeriodMatch | None:
    if not text:
        return None
    text_norm = text.strip()

    match = _MONTH_RANGE_PATTERN.search(text_norm)
    if match:
        start = _month_year_to_date(match.group("m1"), match.group("y1"), day=1)
        end = _month_year_to_date(match.group("m2"), match.group("y2"), day=1, end_of_month=True)
        label = f"{match.group('m1')} {match.group('y1')} - {match.group('m2')} {match.group('y2')}"
        return _PeriodMatch(label=label, start=start, end=end, source=match.group(0))

    match = _HALF_PATTERN.search(text_norm) or _HALF_PATTERN_REV.search(text_norm)
    if match:
        half = match.group("half")
        year = _parse_year(match.group("year"))
        label = f"H{half} FY{str(year)[-2:]}"
        start, end = _half_year_bounds(year, int(half))
        return _PeriodMatch(label=label, start=start, end=end, source=match.group(0))

    match = _QUARTER_PATTERN.search(text_norm) or _QUARTER_PATTERN_REV.search(text_norm)
    if match:
        quarter = match.group("quarter")
        year = _parse_year(match.group("year"))
        label = f"Q{quarter} {year}"
        start, end = _quarter_bounds(year, int(quarter))
        return _PeriodMatch(label=label, start=start, end=end, source=match.group(0))

    match = _FY_PATTERN.search(text_norm)
    if match:
        year = _parse_year(match.group("year"))
        label = f"FY{str(year)[-2:]}"
        start, end = _fiscal_year_bounds(year)
        return _PeriodMatch(label=label, start=start, end=end, source=match.group(0))

    lowered = text_norm.lower()
    for key, label in _RELATIVE_PERIODS.items():
        if key in lowered:
            return _PeriodMatch(label=label, start=None, end=None, source=key)

    return None


def _extract_baseline(text: str) -> str | None:
    if not text:
        return None
    for pattern in _BASELINE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            if match.groups():
                return f"vs {match.group(1).strip()}"
            return match.group(0).strip()
    return None


def _join_text(*parts: str | None) -> str:
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def _format_document_context(document_context: DocumentContext) -> str:
    return (
        f"Client: {document_context.client_name or 'Unknown'}, "
        f"Period: {document_context.report_period or 'Unknown'}"
    )


def _parse_year(value: str) -> int:
    year = int(value)
    if year < 100:
        return 2000 + year
    return year


def _month_year_to_date(month: str, year: str, *, day: int = 1, end_of_month: bool = False) -> str | None:
    month_num = _MONTH_MAP.get(month.lower())
    if not month_num:
        return None
    year_num = int(year)
    if end_of_month:
        end_day = _end_of_month(year_num, month_num)
        return date(year_num, month_num, end_day).isoformat()
    return date(year_num, month_num, day).isoformat()


def _quarter_bounds(fiscal_year: int, quarter: int) -> tuple[str | None, str | None]:
    start_month = _fiscal_start_month()
    fiscal_start_year = fiscal_year if start_month == 1 else fiscal_year - 1
    offset_months = (quarter - 1) * 3
    start_year, start_mon = _add_months(fiscal_start_year, start_month, offset_months)
    end_year, end_mon = _add_months(start_year, start_mon, 2)
    start = date(start_year, start_mon, 1)
    end_day = _end_of_month(end_year, end_mon)
    end = date(end_year, end_mon, end_day)
    return start.isoformat(), end.isoformat()


def _half_year_bounds(fiscal_year: int, half: int) -> tuple[str | None, str | None]:
    start_month = _fiscal_start_month()
    fiscal_start_year = fiscal_year if start_month == 1 else fiscal_year - 1
    offset_months = (half - 1) * 6
    start_year, start_mon = _add_months(fiscal_start_year, start_month, offset_months)
    end_year, end_mon = _add_months(start_year, start_mon, 5)
    start = date(start_year, start_mon, 1)
    end_day = _end_of_month(end_year, end_mon)
    end = date(end_year, end_mon, end_day)
    return start.isoformat(), end.isoformat()


def _fiscal_year_bounds(fiscal_year: int) -> tuple[str | None, str | None]:
    start_month = _fiscal_start_month()
    fiscal_start_year = fiscal_year if start_month == 1 else fiscal_year - 1
    start = date(fiscal_start_year, start_month, 1)
    end_year, end_month = _add_months(fiscal_start_year, start_month, 11)
    end_day = _end_of_month(end_year, end_month)
    end = date(end_year, end_month, end_day)
    return start.isoformat(), end.isoformat()


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = (year * 12 + (month - 1)) + delta
    new_year = total // 12
    new_month = total % 12 + 1
    return new_year, new_month


def _end_of_month(year: int, month: int) -> int:
    if month in {1, 3, 5, 7, 8, 10, 12}:
        return 31
    if month in {4, 6, 9, 11}:
        return 30
    # February
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 29
    return 28
