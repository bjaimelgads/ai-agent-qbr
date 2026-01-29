from types import SimpleNamespace

from qbr_intelligence.metrics.models import MetricAliasSpec, MetricCatalogEntry
from qbr_intelligence.pipeline.metric_scanner import MetricCandidate, build_metric_dictionary
from qbr_intelligence.pipeline.processor import QBRProcessor


def test_extract_context_window_prefers_line_match():
    text = "Header line\nCTR 2.3% vs 1.9%\nFooter"
    snippet = QBRProcessor._extract_context_window(text, ["CTR", "2.3%"])
    assert snippet == "Header line CTR 2.3% vs 1.9% Footer"


def test_build_metric_review_context_includes_sources():
    metric = MetricCandidate(
        metric_id="m1",
        name="Click Through Rate",
        raw_value="2.3%",
        normalized_value=2.3,
        unit="percent",
        raw_context="CTR up QoQ",
        slide_number=1,
        metric_type="percent",
        source="slide_text",
        category="performance",
    )
    slide = {
        "slide_number": 1,
        "raw_text": "Performance\nCTR 2.3% vs 1.9%\nMore text",
        "speaker_notes": "Notes mention CTR 2.3% target",
    }
    catalog_entry = MetricCatalogEntry(
        metric_id="click_through_rate",
        name="Click Through Rate",
        aliases=(MetricAliasSpec(alias="CTR"),),
        expected_unit="percent",
    )
    context = QBRProcessor._build_metric_review_context(
        metric=metric,
        slide=slide,
        catalog_entry=catalog_entry,
    )
    assert "raw_context:" in context
    assert "slide_text:" in context
    assert "speaker_notes:" in context


def test_refine_metrics_with_llm_uses_metric_groups(monkeypatch, tmp_path):
    class DummyReviewer:
        def __init__(self, lm=None):
            pass

        def __call__(self, metric_catalog, context_snippets, candidates):
            return SimpleNamespace(
                review=SimpleNamespace(
                    chosen_index=1,
                    normalized_value=123.0,
                    unit="currency",
                    notes="ok",
                    confidence=0.9,
                    source_snippet="picked",
                )
            )

    monkeypatch.setattr(
        "qbr_intelligence.pipeline.processor.MetricReviewer", DummyReviewer
    )

    processor = QBRProcessor(
        database_url="sqlite:///:memory:",
        output_dir=tmp_path,
        llm_model="openai/gpt-4o-mini",
    )

    metrics = [
        MetricCandidate(
            metric_id="m1",
            name="Cost per Click",
            raw_value="$1.20",
            normalized_value=1.2,
            unit="currency",
            raw_context="CPC $1.20",
            slide_number=1,
            metric_type="currency",
            source="slide_text",
            category="cost",
            extraction_confidence=0.4,
        ),
        MetricCandidate(
            metric_id="m2",
            name="Cost per Click",
            raw_value="$1.10",
            normalized_value=1.1,
            unit="currency",
            raw_context="CPC $1.10",
            slide_number=1,
            metric_type="currency",
            source="slide_text",
            category="cost",
            extraction_confidence=0.5,
        ),
    ]
    slides = [{"slide_number": 1, "raw_text": "CPC $1.10", "speaker_notes": ""}]
    refined = processor._refine_metrics_with_llm(
        metrics=metrics,
        slides=slides,
        metric_dictionary=build_metric_dictionary(),
    )

    assert len(refined) == 1
    assert refined[0].normalized_value == 123.0
    assert refined[0].unit == "currency"
    assert refined[0].metadata["llm_review"]["chosen_index"] == 1
