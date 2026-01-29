from qbr_intelligence.pipeline.metric_scanner import MetricCandidate
from qbr_intelligence.pipeline.processor import QBRProcessor


def test_infer_document_region_from_title(tmp_path):
    processor = QBRProcessor(
        database_url="sqlite:///:memory:",
        output_dir=tmp_path,
        llm_model="openai/gpt-4o-mini",
    )
    region_id, country = processor._infer_document_region("Disney+ US FY24 H2")
    assert region_id == processor._region_map.get("US")
    assert country == "United States"


def test_infer_metric_region_from_country_context(tmp_path):
    processor = QBRProcessor(
        database_url="sqlite:///:memory:",
        output_dir=tmp_path,
        llm_model="openai/gpt-4o-mini",
    )
    metric = MetricCandidate(
        metric_id="m1",
        name="Impressions",
        raw_value="10M",
        normalized_value=10_000_000,
        unit="count",
        raw_context="Germany performance highlights",
        slide_number=1,
        metric_type="count",
        source="slide_text",
        category="performance",
        metadata={},
    )
    region_id, country = processor._infer_metric_region(
        metric, document_region_id=processor._region_map.get("US")
    )
    assert region_id == processor._region_map.get("EMEA")
    assert country == "Germany"


def test_infer_metric_region_from_geo_qualifier(tmp_path):
    processor = QBRProcessor(
        database_url="sqlite:///:memory:",
        output_dir=tmp_path,
        llm_model="openai/gpt-4o-mini",
    )
    metric = MetricCandidate(
        metric_id="m1",
        name="Reach",
        raw_value="5M",
        normalized_value=5_000_000,
        unit="count",
        raw_context="EMEA Reach",
        slide_number=2,
        metric_type="count",
        source="slide_text",
        category="reach",
        metadata={"qualifiers": {"geo": "EMEA"}},
    )
    region_id, country = processor._infer_metric_region(
        metric, document_region_id=processor._region_map.get("US")
    )
    assert region_id == processor._region_map.get("EMEA")
    assert country is None


def test_infer_metric_region_prefers_country_over_raw_context_us(tmp_path):
    processor = QBRProcessor(
        database_url="sqlite:///:memory:",
        output_dir=tmp_path,
        llm_model="openai/gpt-4o-mini",
    )
    metric = MetricCandidate(
        metric_id="m1",
        name="Impressions",
        raw_value="1M",
        normalized_value=1_000_000,
        unit="count",
        raw_context="Markets: US, UK, ES",
        slide_number=3,
        metric_type="count",
        source="slide_text",
        category="performance",
        metadata={},
    )
    region_id, country = processor._infer_metric_region(
        metric, document_region_id=processor._region_map.get("US")
    )
    assert region_id == processor._region_map.get("EMEA")
    assert country == "United Kingdom"
