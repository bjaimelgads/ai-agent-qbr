from __future__ import annotations

from qbr_intelligence.schemas.metric_qa import QueryIntent
from qbr_intelligence.metric_qa.intent import classify_intent_type


def test_intent_type_definition():
    intent = QueryIntent(metric_ids=["cpa"])
    assert classify_intent_type("What is CPA?", intent) == "DEFINITION"


def test_intent_type_analytics():
    intent = QueryIntent(metric_ids=["cpa"], aggregation="trend")
    assert classify_intent_type("Show CPA trend", intent) == "ANALYTICS"
