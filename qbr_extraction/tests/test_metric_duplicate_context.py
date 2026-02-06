from qbr_intelligence.pipeline.metric_scanner import MetricCandidate
from qbr_intelligence.pipeline.processor import QBRProcessor


def test_augment_duplicate_metric_context_expands_snippet():
    metrics = [
        MetricCandidate(
            metric_id="m1",
            name="Cost per Acquisition",
            raw_value="$12.00",
            normalized_value=12.0,
            unit="currency",
            raw_context="CPA $12.00",
            slide_number=1,
            metric_type="currency",
            source="slide_text",
            category="cost",
        ),
        MetricCandidate(
            metric_id="m2",
            name="Cost per Acquisition",
            raw_value="$12.00",
            normalized_value=12.0,
            unit="currency",
            raw_context="CPA $12.00",
            slide_number=2,
            metric_type="currency",
            source="slide_text",
            category="cost",
        ),
    ]
    slides = [
        {"slide_number": 1, "raw_text": "US Performance\nCPA $12.00 for Q1"},
        {"slide_number": 2, "raw_text": "EMEA Performance\nCPA $12.00 for Q1"},
    ]

    updated = QBRProcessor._augment_duplicate_metric_context(metrics, slides)

    assert len(updated) == 2
    assert "slide 1:" in updated[0].raw_context
    assert "US Performance" in updated[0].raw_context
    assert "slide 2:" in updated[1].raw_context
    assert "EMEA Performance" in updated[1].raw_context
