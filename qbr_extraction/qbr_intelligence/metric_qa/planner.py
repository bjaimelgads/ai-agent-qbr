"""Query planning for metric QA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from qbr_intelligence.schemas.metric_qa import QueryIntent


DEFAULT_LIMIT = 200


@dataclass
class MetricQueryPlan:
    metric_ids: list[str]
    client: str | None
    region: str | None
    period_start: date | None
    period_end: date | None
    limit: int
    order_by: str
    aggregation: str | None
    grouping: str | None


def build_plan(intent: QueryIntent) -> MetricQueryPlan:
    limit = intent.limit or DEFAULT_LIMIT
    order_by = "period_end DESC"
    if intent.aggregation == "trend":
        order_by = "period_end ASC"
    return MetricQueryPlan(
        metric_ids=intent.metric_ids,
        client=intent.client,
        region=intent.region,
        period_start=intent.period.start if intent.period else None,
        period_end=intent.period.end if intent.period else None,
        limit=limit,
        order_by=order_by,
        aggregation=intent.aggregation,
        grouping=intent.grouping,
    )
