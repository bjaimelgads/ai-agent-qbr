import os

os.environ.setdefault("QBR_INTELLIGENCE_LIGHT_IMPORT", "1")

from qbr_intelligence.pipeline.metric_context import (
    DocumentContext,
    HybridMetricContextStrategy,
    LLMMetricsContextStrategy,
    RuleBasedMetricContextStrategy,
    apply_context_strategies,
)
from qbr_intelligence.pipeline.metric_scanner import MetricCandidate


class SimpleContext:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class SimpleOutput:
    def __init__(self, contexts):
        self.contexts = contexts


class FakeExtractor:
    def __init__(self, *, brand: str | None = None, period_label: str | None = None):
        self.brand = brand
        self.period_label = period_label

    def __call__(self, slide_text: str, speaker_notes: str, candidates: list[dict], document_context: str):
        metric_id = candidates[0]["id"]
        return SimpleOutput(
            contexts=[
                SimpleContext(
                    metric_id=metric_id,
                    period_label=self.period_label,
                    brand=self.brand,
                    baseline_text="YoY",
                    baseline_type="yoy",
                    source_snippet="Q1 2025 YoY",
                )
            ]
        )


def _make_metric(metric_id: str, raw_context: str, slide_number: int | None = 1) -> MetricCandidate:
    return MetricCandidate(
        metric_id=metric_id,
        name="Spend",
        raw_value="$2.8M",
        normalized_value=2800000.0,
        unit="currency",
        raw_context=raw_context,
        slide_number=slide_number,
        metric_type="currency",
        source="slides_parsed",
        category="cost",
    )


def test_rule_based_strategy_extracts_period_brand_baseline():
    metrics = [_make_metric("m1", "H2 FY25 investment, 8% increase vs last quarter", 22)]
    slides = [{"slide_number": 22, "raw_text": "Brand: Disney+ | H2 FY25", "speaker_notes": ""}]
    document_context = DocumentContext(client_name="Disney+", report_period="FY25 H2")
    strategies = [RuleBasedMetricContextStrategy()]

    updated = apply_context_strategies(
        metrics=metrics,
        slides=slides,
        document_context=document_context,
        strategies=strategies,
        stage="scanned",
    )

    assert updated[0].period_label == "H2 FY25"
    assert updated[0].period_start == "2025-04-01"
    assert updated[0].period_end == "2025-09-30"
    assert updated[0].brand == "Disney+"
    assert updated[0].baseline_type == "qoq"


def test_llm_strategy_applies_context():
    metrics = [_make_metric("m2", "Q1 2025 performance", 5)]
    slides = [{"slide_number": 5, "raw_text": "Q1 2025 results", "speaker_notes": ""}]
    document_context = DocumentContext(client_name="Hulu", report_period=None)
    strategies = [LLMMetricsContextStrategy(FakeExtractor(brand="Hulu", period_label="Q1 2025"))]

    updated = apply_context_strategies(
        metrics=metrics,
        slides=slides,
        document_context=document_context,
        strategies=strategies,
        stage="refined",
    )

    assert updated[0].period_label == "Q1 2025"
    assert updated[0].brand == "Hulu"
    assert updated[0].baseline_type == "yoy"


def test_hybrid_strategy_fills_missing_fields():
    metrics = [_make_metric("m3", "H1 FY25 spend grew", 7)]
    slides = [{"slide_number": 7, "raw_text": "H1 FY25 results", "speaker_notes": ""}]
    document_context = DocumentContext(client_name=None, report_period=None)
    rule_strategy = RuleBasedMetricContextStrategy()
    llm_strategy = LLMMetricsContextStrategy(FakeExtractor(brand="Disney+", period_label=None))
    strategies = [
        HybridMetricContextStrategy(rule_strategy=rule_strategy, llm_strategy=llm_strategy)
    ]

    updated = apply_context_strategies(
        metrics=metrics,
        slides=slides,
        document_context=document_context,
        strategies=strategies,
        stage="refined",
    )

    assert updated[0].period_label == "H1 FY25"
    assert updated[0].period_start == "2024-10-01"
    assert updated[0].period_end == "2025-03-31"
    assert updated[0].brand == "Disney+"
