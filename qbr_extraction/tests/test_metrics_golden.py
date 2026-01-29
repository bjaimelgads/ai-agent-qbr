from __future__ import annotations

import json
from pathlib import Path

import pytest

from qbr_intelligence.metrics.pipeline import MetricExtractionPipeline

DECKS_DIR = Path(__file__).resolve().parents[1] / "decks"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def _load_metrics(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(metrics: list[dict]) -> list[dict]:
    for metric in metrics:
        if metric.get("value") is not None:
            metric["value"] = round(float(metric["value"]), 6)
        metric["confidence"] = round(float(metric.get("confidence", 0.0)), 4)
    metrics.sort(
        key=lambda m: (
            m.get("slide_index"),
            m.get("metric_id"),
            m.get("raw_value_text"),
            m.get("label_text"),
        )
    )
    return metrics


@pytest.mark.parametrize(
    "deck_filename",
    [
        "Disney+ US FY24 H1.pptx",
        "Disney+ US FY24 H2.pptx",
        "Disney+ US FY25 H1.pptx",
        "Disney+ US FY25 H2.pptx",
    ],
)

def test_metrics_against_golden(deck_filename: str) -> None:
    deck_path = DECKS_DIR / deck_filename
    golden_path = GOLDEN_DIR / f"{deck_filename}.metrics.json"
    assert deck_path.exists(), f"Missing deck: {deck_path}"
    assert golden_path.exists(), f"Missing golden: {golden_path}"

    pipeline = MetricExtractionPipeline()
    metrics, _ = pipeline.extract_from_pptx(deck_path)
    actual = _normalize([m.to_dict() for m in metrics])
    expected = _normalize(_load_metrics(golden_path))

    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        assert act["slide_index"] == exp["slide_index"]
        assert act["metric_id"] == exp["metric_id"]
        assert act["unit"] == exp["unit"]
        assert act["raw_value_text"] == exp["raw_value_text"]
        assert act["label_text"] == exp["label_text"]
        assert abs(act["value"] - exp["value"]) <= 1e-4
