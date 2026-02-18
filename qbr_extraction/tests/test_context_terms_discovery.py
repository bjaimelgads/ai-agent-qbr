from __future__ import annotations

from pathlib import Path

from qbr_intelligence.pipeline.context_terms_discovery import (
    CatalogMatcher,
    _table_dimension_mentions,
    discover_metric_context_terms,
    is_metric_or_noise_term,
)


def test_metric_tokens_are_filtered_from_context_terms() -> None:
    matcher = CatalogMatcher(exact_labels={"ctr", "click through rate", "impressions"}, regex_patterns=tuple())

    assert is_metric_or_noise_term("ctr", matcher=matcher) is True
    assert is_metric_or_noise_term("impressions", matcher=matcher) is True
    assert is_metric_or_noise_term("not installed", matcher=matcher) is False


def test_table_dimension_extraction_finds_non_metric_series_terms() -> None:
    matcher = CatalogMatcher(exact_labels={"ctr", "click through rate", "impressions"}, regex_patterns=tuple())
    metric_phrases = {
        "ctr": "click_through_rate",
        "impressions": "impressions",
    }
    rows = [
        ["Metric", "Not Installed", "Installed", "Roadblock"],
        ["CTR", "0.11%", "0.14%", "0.31%"],
        ["Impressions", "1.2M", "2.8M", "6.0M"],
    ]

    mentions = _table_dimension_mentions(
        rows=rows,
        deck="Deck A",
        slide_number=9,
        metric_phrases=metric_phrases,
        matcher=matcher,
    )
    terms = {m.term for m in mentions}

    assert "not installed" in terms
    assert "installed" in terms
    assert "roadblock" in terms
    assert "ctr" not in terms


def test_discover_metric_context_terms_on_sample_box_file(tmp_path: Path) -> None:
    root = tmp_path / "output"
    deck_dir = root / "Sample Deck"
    deck_dir.mkdir(parents=True)
    box_file = deck_dir / "02b_raw_content_by_box.txt"
    box_file.write_text(
        """
<!-- PAGE 1 -->
- - - -
[TABLE]
Metric | Not Installed | Installed | Roadblock
CTR | 0.11% | 0.14% | 0.31%
Impressions | 1.2M | 2.8M | 6.0M
- - - -

- - - -
Video Enabled Roadblocks driving CTR up to 2x higher vs static carousel.
- - - -
""".strip(),
        encoding="utf-8",
    )

    contexts, by_metric = discover_metric_context_terms(
        root=root,
        database_url="sqlite+aiosqlite:///nonexistent_test.db",
        min_occurrences=1,
    )

    context_terms = {row["context_term"] for row in contexts}
    assert "not installed" in context_terms
    assert "installed" in context_terms
    assert "roadblock" in context_terms or "roadblocks" in context_terms

    by_metric_terms = {row["context_term"] for row in by_metric if row["metric_id"] == "click_through_rate"}
    assert "not installed" in by_metric_terms
