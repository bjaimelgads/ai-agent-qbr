from __future__ import annotations

from qbr_intelligence.metric_qa.composer import AnswerComposer
from qbr_intelligence.metric_qa.dao import MetricFactRow
from qbr_intelligence.schemas.metric_qa import QueryIntent


def _row(value: float, period_end: str, doc_id: int, slide_id: int) -> MetricFactRow:
    return MetricFactRow(
        fact_id=1,
        metric_id="cpa",
        metric_name="CPA",
        value=value,
        unit="currency",
        scale="ones",
        client_id=1,
        client_name="Nike",
        region="US",
        period_start="2025-01-01",
        period_end=period_end,
        period_granularity="quarter",
        period_label=None,
        document_id=doc_id,
        slide_id=slide_id,
        confidence=0.9,
        label_text="CPA",
        raw_value_text=None,
        snippet="snippet",
    )


def test_compare_aggregation_includes_delta():
    rows = [_row(10.0, "2025-03-31", 10, 3), _row(12.0, "2025-06-30", 10, 4)]
    intent = QueryIntent(metric_ids=["cpa"], aggregation="compare")
    answer = AnswerComposer().compose(intent=intent, rows=rows, assumptions=[])
    assert "Change" in answer.summary_text
    assert answer.citations
