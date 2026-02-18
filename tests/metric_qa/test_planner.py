from __future__ import annotations

from qbr_intelligence.schemas.metric_qa import QueryIntent
from qbr_intelligence.metric_qa.planner import build_plan


def test_build_plan_defaults():
    intent = QueryIntent(metric_ids=["cpa"])
    plan = build_plan(intent)
    assert plan.limit == 200
    assert plan.order_by == "period_end DESC"


def test_build_plan_trend_order():
    intent = QueryIntent(metric_ids=["ctr"], aggregation="trend")
    plan = build_plan(intent)
    assert plan.order_by == "period_end ASC"
