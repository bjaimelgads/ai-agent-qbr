from __future__ import annotations

from qbr_intelligence.metrics.models import MetricExtraction
from qbr_intelligence.metrics.resolver import resolve_metrics


def _metric(metric_id: str, label_text: str) -> MetricExtraction:
    return MetricExtraction(
        deck_id="deck",
        slide_index=1,
        metric_id=metric_id,
        metric_name="Reach" if metric_id == "reach" else "Unique Reach",
        value=100.0,
        unit="count",
        scale="ones",
        raw_value_text="100",
        label_text=label_text,
        qualifiers={},
        confidence=0.7,
        extraction_method="same_block",
        provenance={"source_type": "slide_text", "shape_id_or_cell_id": "1", "snippet": label_text},
    )


def test_resolve_unique_reach() -> None:
    metrics = [_metric("reach", "Deduplicated reach")]
    resolved = resolve_metrics(metrics)
    assert resolved[0].metric_id == "unique_reach"


def test_drop_redundant_reach() -> None:
    metrics = [_metric("reach", "Reach"), _metric("unique_reach", "Unique Reach")]
    resolved = resolve_metrics(metrics)
    assert len(resolved) == 1
    assert resolved[0].metric_id == "unique_reach"
