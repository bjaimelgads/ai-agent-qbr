from qbr_intelligence.metric_qa.composer import AnswerComposer
from qbr_intelligence.metric_qa.dao import MetricFactRow
from qbr_intelligence.schemas.metric_qa import QueryIntent


def test_metric_answer_includes_llm_context_label():
    row = MetricFactRow(
        fact_id=1,
        metric_id="cost_per_acquisition",
        metric_name="Cost per Acquisition",
        value=12.34,
        unit="currency",
        scale="ones",
        client_id=1,
        client_name="Disney+",
        region="US",
        period_start=None,
        period_end=None,
        period_granularity=None,
        period_label="H1 FY24",
        document_id=1,
        document_name="Disney+ US FY24 H1.pptx",
        document_url="https://example.com/deck",
        slide_id=10,
        slide_number=28,
        slide_title="Install Trends",
        google_slide_id=None,
        confidence=0.9,
        label_text="Cost per Acquisition",
        raw_value_text="$12.34",
        snippet="CPA $12.34",
        llm_context_label="US H1 FY24 Roadblock",
    )
    intent = QueryIntent(metric_ids=["cost_per_acquisition"])
    answer = AnswerComposer().compose(intent=intent, rows=[row], assumptions=[])

    assert answer.data is not None
    assert answer.data[0].llm_context_label == "US H1 FY24 Roadblock"
    assert answer.table_data is None or answer.table_data[0].get("llm_context_label") == "US H1 FY24 Roadblock"
