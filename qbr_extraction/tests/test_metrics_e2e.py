from __future__ import annotations

from pathlib import Path

from qbr_intelligence.metrics.pipeline import MetricExtractionPipeline

DECK_PATH = Path(__file__).resolve().parents[1] / "decks" / "Disney+ US FY24 H2.pptx"


def test_metrics_pipeline_deterministic() -> None:
    pipeline = MetricExtractionPipeline()
    metrics_a, _ = pipeline.extract_from_pptx(DECK_PATH)
    metrics_b, _ = pipeline.extract_from_pptx(DECK_PATH)

    a = [m.to_dict() for m in metrics_a]
    b = [m.to_dict() for m in metrics_b]

    assert len(a) == len(b)
    assert a == b


def test_metrics_confidence_distribution() -> None:
    pipeline = MetricExtractionPipeline()
    metrics, _ = pipeline.extract_from_pptx(DECK_PATH)
    if not metrics:
        raise AssertionError("No metrics extracted")
    avg_conf = sum(m.confidence for m in metrics) / len(metrics)
    assert 0.3 <= avg_conf <= 0.95
