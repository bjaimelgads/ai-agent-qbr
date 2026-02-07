"""Query planning for metric QA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from qbr_intelligence.schemas.metric_qa import QueryIntent


DEFAULT_LIMIT = 200


@dataclass
class MetricQueryPlan:
    metric_ids: list[str]
    client: list[str]
    region: list[str]
    period_ranges: list[tuple[date, date]]
    limit: int
    order_by: str
    aggregation: str | None
    grouping: str | None


def build_plan(intent: QueryIntent) -> MetricQueryPlan:
    limit = intent.limit or DEFAULT_LIMIT
    order_by = "period_end DESC"
    if intent.aggregation == "trend":
        order_by = "period_end ASC"
    ranges: list[tuple[date, date]] = []
    for period in intent.period:
        if period.start and period.end:
            ranges.append((period.start, period.end))
    return MetricQueryPlan(
        metric_ids=intent.metric_ids,
        client=intent.client,
        region=intent.region,
        period_ranges=ranges,
        limit=limit,
        order_by=order_by,
        aggregation=intent.aggregation,
        grouping=intent.grouping,
    )
