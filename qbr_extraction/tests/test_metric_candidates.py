from __future__ import annotations

from qbr_intelligence.metrics.candidates import extract_value_candidates, extract_label_candidates


def test_extract_value_candidates_percent_and_currency() -> None:
    text = "CTR: 2.5% | Spend $1.2M"
    values = extract_value_candidates(text)
    raw_values = {v.raw_value_text for v in values}
    assert "2.5%" in raw_values
    assert any(v.raw_value_text.startswith("$1.2") for v in values)


def test_extract_value_candidates_kmb() -> None:
    text = "Reach 120K, Impressions 3.4M, Budget $2B"
    values = extract_value_candidates(text)
    norm = {v.normalized_value for v in values}
    assert 120_000 in norm
    assert 3_400_000 in norm
    assert 2_000_000_000 in norm


def test_extract_value_candidates_ignore_years() -> None:
    text = "FY24 performance in 2024 shows reach 1.2M"
    values = extract_value_candidates(text)
    raw_values = {v.raw_value_text for v in values}
    assert "2024" not in raw_values
    assert "1.2M" in raw_values


def test_extract_label_candidates() -> None:
    text = "CTR and conversion rate are up"
    labels = extract_label_candidates(text, source_type="slide_text", block_id="b1", slide_index=1)
    metric_ids = {label.metric_id for label in labels}
    assert "click_through_rate" in metric_ids
    assert "conversion_rate" in metric_ids
