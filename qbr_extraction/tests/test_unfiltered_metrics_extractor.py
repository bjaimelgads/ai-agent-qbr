from __future__ import annotations

import uuid
from pathlib import Path

from qbr_intelligence.pipeline.unfiltered_metrics_extractor import UnfilteredMetricsExtractor


def _write_02b(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "02b_raw_content_by_box.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_slide7_country_rows_resolve_to_descriptor(tmp_path: Path) -> None:
    content = """
<!-- PAGE 7 -->
- - - -
LG TV Footprint by Market
- - - -

- - - -
[TABLE]
7.85M | United Kingdom
5.38M | Spain
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    assert len(records) == 2
    assert all(r.name == "LG TV Footprint" for r in records)
    countries = {(r.metadata or {}).get("country") for r in records}
    assert countries == {"United Kingdom", "Spain"}


def test_table_column_metric_orientation_extracts_metric_names(tmp_path: Path) -> None:
    content = """
<!-- PAGE 67 -->
- - - -
Overall Performance By Targeting Strategy
- - - -

- - - -
[TABLE]
 | Install Rate | CPE* | CPA*
Disney+ Lapsed | N/A | $3.30 | N/A
Disney+ Not Installed | 0.57% | $20.5 | $17.80
Untargeted RB | 0.08% | $0.6 | $110.4
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    names = {r.name for r in records}
    assert "Install Rate" in names
    assert "CPE" in names
    assert "CPA" in names
    assert "Disney+ Not Installed" not in names
    assert "Untargeted RB" not in names
    cpe = [r for r in records if r.name == "CPE" and r.raw_value == "$20.5"]
    assert len(cpe) == 1
    uuid.UUID(cpe[0].table_id or "")
    assert cpe[0].metadata.get("table_id") == cpe[0].table_id
    assert cpe[0].metadata.get("table_context_label") == "Disney+ Not Installed"
    assert "Overall Performance By Targeting Strategy" in cpe[0].raw_context


def test_table_row_metric_orientation_extracts_all_series_values(tmp_path: Path) -> None:
    content = """
<!-- PAGE 69 -->
- - - -
RBs Boosts Engagement, ROS Drives Acquisitions
- - - -

- - - -
[TABLE]
 | Rotational | Roadblocks
Install Rate* | 0.60% | 0.08%
CPA | $14.06 | $110.42
CTR | 0.11% | 0.14%
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    ctr = [r for r in records if r.name == "CTR"]
    assert len(ctr) == 2
    contexts = {r.metadata.get("table_context_label") for r in ctr}
    assert contexts == {"Rotational", "Roadblocks"}
    assert all(r.table_id for r in ctr)
    assert len({r.table_id for r in ctr}) == 1
    assert all("RBs Boosts Engagement, ROS Drives Acquisitions" in r.raw_context for r in ctr)


def test_metric_name_sanitization_removes_trailing_footnote_markers(tmp_path: Path) -> None:
    content = """
<!-- PAGE 1 -->
- - - -
[TABLE]
 | Rotational | Roadblocks
Install Rate* | 0.60% | 0.08%
CPA† | $14.06 | $110.42
CPE[1] | $6.21 | $0.56
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)
    names = {r.name for r in records}
    assert "Install Rate" in names
    assert "CPA" in names
    assert "CPE" in names
    assert "Install Rate*" not in names
    assert "CPA†" not in names
    assert "CPE[1]" not in names


def test_at_a_glance_slide_marks_overall_metrics(tmp_path: Path) -> None:
    content = """
<!-- PAGE 24 -->
- - - -
EMEA FY24 H2 at a Glance
- - - -

- - - -
Total Installs:
699K
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)
    assert len(records) == 1
    rec = records[0]
    assert rec.name == "Total Installs"
    assert rec.metadata.get("is_overall_metric") is True
    assert rec.metadata.get("overall_slide_number") == 24
    assert rec.metadata.get("overall_slide_title") == "EMEA FY24 H2 at a Glance"


def test_fiscal_title_line_does_not_become_metric_and_marks_overall(tmp_path: Path) -> None:
    content = """
<!-- PAGE 21 -->
- - - -
FY25 - H2
at a Glance
- - - -

- - - -
Investment
$2.8M
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(
        review_threshold=0.65,
        catalog_hint_terms={"investment"},
    )
    records = extractor.extract(path)
    names = {r.name for r in records}
    assert "FY25" not in names
    assert "Investment" in names
    inv = next(r for r in records if r.name == "Investment")
    assert inv.metadata.get("is_overall_metric") is True
    assert inv.metadata.get("overall_slide_number") == 21


def test_value_leading_single_line_cards_are_extracted(tmp_path: Path) -> None:
    content = """
<!-- PAGE 22 -->
- - - -
Proxy Conversions Funnel
- - - -

- - - -
$2.8M Investment
- - - -

- - - -
308M Impressions
- - - -

- - - -
17M Total Unique Reach
- - - -

- - - -
1.6M Proxy Conversions
- - - -

- - - -
$1.79 Cost per Proxy Conversion
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65, catalog_hint_terms={"investment", "impressions"})
    records = extractor.extract(path)
    got = {(r.name, r.raw_value) for r in records}
    assert ("Investment", "$2.8M") in got
    assert ("Impressions", "308M") in got
    assert ("Total Unique Reach", "17M") in got
    assert ("Proxy Conversions", "1.6M") in got
    assert ("Cost per Proxy Conversion", "$1.79") in got


def test_catalog_hint_terms_prevent_metric_hint_penalty(tmp_path: Path) -> None:
    content = """
<!-- PAGE 1 -->
- - - -
Investment: $2.25M
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(
        review_threshold=0.65,
        catalog_hint_terms={"investment"},
    )
    records = extractor.extract(path)
    assert len(records) == 1
    name_components = records[0].metadata["confidence_components"]["name"]
    assert "metric_hint_penalty" not in name_components
    assert name_components.get("metric_hint_bonus") == 0.06


def test_slide19_country_growth_block_extracts_values(tmp_path: Path) -> None:
    content = """
<!-- PAGE 19 -->
- - - -
D+ Growth in MAUs (April -> Sept '24)
- - - -

- - - -
Greece: 133K   +11%
Turkey: 262K   +10%
Spain: 1.2M    +8%
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    assert len(records) == 3
    assert all(r.name == "D+ Growth in MAUs (April -> Sept '24)" for r in records)
    values = sorted(r.raw_value for r in records)
    assert values == ["1.2M", "133K", "262K"]


def test_slide20_cross_box_patterns_are_extracted(tmp_path: Path) -> None:
    content = """
<!-- PAGE 20 -->
- - - -
Avg duration per launch
- - - -

- - - -
56.4 mins
- - - -

- - - -
64.7mins
- - - -

- - - -
+15%
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    assert len(records) == 3
    assert all(r.name == "Avg duration per launch" for r in records)
    assert any(r.unit == "time" for r in records)
    assert any(r.unit == "percent" for r in records)


def test_confidence_and_review_flag_present(tmp_path: Path) -> None:
    content = """
<!-- PAGE 1 -->
- - - -
Investment: $2.25M
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.99)
    records = extractor.extract(path)

    assert len(records) == 1
    rec = records[0]
    assert rec.extraction_confidence is not None
    assert 0.0 <= rec.extraction_confidence <= 1.0
    assert isinstance(rec.metadata, dict)
    assert "confidence_components" in rec.metadata
    assert rec.metadata.get("review_recommended") is True


def test_raw_context_selection_metadata_present(tmp_path: Path) -> None:
    content = """
<!-- PAGE 20 -->
- - - -
Avg duration per launch
- - - -

- - - -
56.4 mins
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    assert len(records) == 1
    rec = records[0]
    assert "Avg duration per launch" in rec.raw_context
    assert "56.4 mins" in rec.raw_context
    assert isinstance(rec.metadata, dict)
    assert rec.metadata.get("context_source") in {"descriptor_plus_box", "box", "nearby_plus_box"}
    assert isinstance(rec.metadata.get("context_confidence"), float)
    assert isinstance(rec.metadata.get("context_candidates"), list)


def test_metric_name_confidence_penalizes_long_singletons(tmp_path: Path) -> None:
    content = """
<!-- PAGE 1 -->
- - - -
CPA
$12.5
- - - -

- - - -
CPA
$10.2
- - - -

- - - -
Exposure to paid media led to an average time spent per launch of over one hour versus the overall audience
56.4 mins
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    cpa = [r for r in records if r.name == "CPA"]
    long_name = [
        r
        for r in records
        if r.name.startswith("Exposure to paid media led to an average time spent per launch")
    ]
    assert len(cpa) >= 2
    assert len(long_name) == 1

    cpa_conf = cpa[0].metadata["confidence_components"]["name"]["final"]
    long_conf = long_name[0].metadata["confidence_components"]["name"]["final"]
    assert long_conf < cpa_conf
    long_blend = long_name[0].metadata["confidence_components"]["blend"]
    assert "penalty_total" in long_blend
    assert (
        "long_name_penalty" in long_blend
        or "very_long_name_penalty" in long_blend
        or "long_token_penalty" in long_blend
        or "very_long_token_penalty" in long_blend
    )


def test_numeric_label_is_not_treated_as_metric_name(tmp_path: Path) -> None:
    content = """
<!-- PAGE 16 -->
- - - -
As viewers navigate down the LG Home Screen to view additional content
Markets:  UK, ES, DE,  IT, US, & AU
TV Models: 2020+
December: 2024,  2023, & 2022
2025: 2021 and older
Ad Format: 1770x221 (static)
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    assert all(r.name != "2025" for r in records)
    assert all(not (r.name == "2025" and r.raw_value == "202") for r in records)


def test_ratio_decimal_value_is_preserved(tmp_path: Path) -> None:
    content = """
<!-- PAGE 24 -->
- - - -
Avg Monthly Freq
6.5x
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)
    assert len(records) == 1
    rec = records[0]
    assert rec.raw_value == "6.5x"
    assert rec.unit == "ratio"
    assert rec.normalized_value == 6.5


def test_slide20_prefers_kpi_descriptor_over_sentence_and_dimension_labels(tmp_path: Path) -> None:
    content = """
<!-- PAGE 20 -->
- - - -
EMEA: Paid Media Improves App Engagement
- - - -

- - - -
Exposure to LG paid media led to an average time spent per launch of over 1 hour versus the overall Disney+ audience across EMEA markets
- - - -

- - - -
+15%
- - - -

- - - -
56.4 mins
- - - -

- - - -
64.7mins
- - - -

- - - -
LG exposed audience
- - - -

- - - -
Overall1 audience
- - - -

- - - -
Avg duration per launch
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)

    time_records = [r for r in records if r.raw_value in {"56.4 mins", "64.7mins"}]
    assert len(time_records) == 2
    assert all(r.name == "Avg duration per launch" for r in time_records)
    assert all(r.unit == "time" for r in time_records)


def test_thin_stacked_box_context_is_enriched_with_descriptor(tmp_path: Path) -> None:
    content = """
<!-- PAGE 25 -->
- - - -
EMEA FY24 H2 Media Strategy by Market
- - - -

- - - -
Roadblocks
10%
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)
    roadblocks = [r for r in records if r.name == "Roadblocks" and r.raw_value == "10%"]
    assert len(roadblocks) == 1
    rec = roadblocks[0]
    assert "EMEA FY24 H2 Media Strategy" in rec.raw_context
    assert rec.metadata.get("context_source") == "descriptor_plus_box"


def test_cross_box_context_prefers_richer_descriptor_when_label_equals_metric(tmp_path: Path) -> None:
    content = """
<!-- PAGE 20 -->
- - - -
EMEA: Paid Media Improves App Engagement
- - - -

- - - -
Exposure to LG paid media led to an average time spent per launch of over 1 hour versus the overall Disney+ audience across EMEA markets
- - - -

- - - -
64.7mins
- - - -

- - - -
Avg duration per launch
- - - -
""".strip()
    path = _write_02b(tmp_path, content)

    extractor = UnfilteredMetricsExtractor(review_threshold=0.65)
    records = extractor.extract(path)
    hits = [r for r in records if r.raw_value == "64.7mins" and r.name == "Avg duration per launch"]
    assert len(hits) == 1
    rec = hits[0]
    assert rec.metadata.get("context_source") in {"richer_descriptor_plus_box", "nearby_plus_box"}
    assert "Avg duration per launch" in rec.raw_context
    assert "64.7mins" in rec.raw_context
    assert len(rec.raw_context.splitlines()) >= 2
