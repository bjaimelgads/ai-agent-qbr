"""Intent extraction for metric QA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Iterable

from qbr_intelligence.schemas.metric_qa import PeriodSpec, QueryIntent
from qbr_intelligence.metrics.models import MetricCatalogEntry

from .resolvers import ClientResolver, MetricResolver, PeriodResolver, RegionResolver
from .utils import normalize_text


@dataclass
class IntentDebug:
    metrics: dict[str, float]
    client: str | None
    client_confidence: float
    region: str | None
    region_confidence: float
    period_label: str | None
    period_type: str | None


class DeterministicIntentExtractor:
    def __init__(
        self,
        metric_catalog: Iterable[MetricCatalogEntry],
        clients: Iterable[str],
    ) -> None:
        self._metric_resolver = MetricResolver(metric_catalog)
        self._client_resolver = ClientResolver(clients)
        self._region_resolver = RegionResolver()
        self._period_resolver = PeriodResolver()

    def extract(self, query: str, *, anchor_date: date | None) -> tuple[QueryIntent, IntentDebug]:
        text = normalize_text(query)
        metric_ids, metric_scores = self._metric_resolver.resolve(text)
        client_res = self._client_resolver.resolve(text)
        region_res = self._region_resolver.resolve(text)
        bounds, label, period_type = self._period_resolver.resolve(text, anchor_date=anchor_date)

        aggregation = _detect_aggregation(text)
        grouping = _detect_grouping(text)
        limit = _detect_limit(text)
        clarifications: list[str] = []
        if not metric_ids:
            clarifications.append("Which metric should I use?")
        if client_res.confidence < 0.5 and client_res.candidates:
            clarifications.append("Which client should I use?")
        if region_res.confidence < 0.5 and region_res.candidates:
            clarifications.append("Which region should I use?")

        period_spec = None
        if bounds:
            period_spec = PeriodSpec(
                type=period_type,
                value=label,
                start=bounds.start,
                end=bounds.end,
            )

        intent = QueryIntent(
            metric_ids=metric_ids,
            client=client_res.value,
            region=region_res.value,
            period=period_spec,
            aggregation=aggregation,
            grouping=grouping,
            limit=limit,
            clarifications_needed=clarifications,
        )
        debug = IntentDebug(
            metrics=metric_scores,
            client=client_res.value,
            client_confidence=client_res.confidence,
            region=region_res.value,
            region_confidence=region_res.confidence,
            period_label=label,
            period_type=period_type,
        )
        return intent, debug


_DEFINITION_PATTERNS = (
    r"\bwhat\s+is\b",
    r"\bdefine\b",
    r"\bdefinition\b",
    r"\bhow\s+is\b",
    r"\bformula\b",
    r"\bcalculated\b",
)


def classify_intent_type(query: str, intent: QueryIntent) -> str:
    text = normalize_text(query)
    if intent.metric_ids and not intent.period and not intent.aggregation:
        if any(re.search(pattern, text) for pattern in _DEFINITION_PATTERNS):
            return "DEFINITION"
    return "ANALYTICS"


_AGGREGATION_MAP = [
    ("latest", ["latest", "most recent", "current", "this quarter", "this month"]),
    ("average", ["average", "avg", "mean"]),
    ("sum", ["total", "sum"]),
    ("min", ["minimum", "lowest", "min"]),
    ("max", ["maximum", "highest", "max"]),
    ("trend", ["trend", "over time", "history", "past"]),
    ("compare", ["compare", "vs", "versus", "difference", "delta"]),
]

_GROUPING_MAP = [
    ("by_period", ["by quarter", "by period", "by month", "by year"]),
    ("by_region", ["by region", "by market"]),
    ("by_client", ["by client", "by advertiser", "by brand"]),
]


def _detect_aggregation(text: str) -> str | None:
    for agg, tokens in _AGGREGATION_MAP:
        if any(token in text for token in tokens):
            return agg
    return None


def _detect_grouping(text: str) -> str | None:
    for grouping, tokens in _GROUPING_MAP:
        if any(token in text for token in tokens):
            return grouping
    return None


def _detect_limit(text: str) -> int | None:
    match = re.search(r"\btop\s+(\d+)\b", text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None
