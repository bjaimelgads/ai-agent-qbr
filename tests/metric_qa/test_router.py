from __future__ import annotations

from ai_agent_qbr.infrastructure.metric_router import MetricQueryRouter


def test_router_detects_metric_alias():
    router = MetricQueryRouter()
    assert router.is_metric_query("What's the CPA for Nike?")


def test_router_detects_time_reference():
    router = MetricQueryRouter()
    assert router.is_metric_query("Compare reach Q1 vs Q2")


def test_router_non_metric():
    router = MetricQueryRouter()
    assert not router.is_metric_query("Summarize the QBR")
