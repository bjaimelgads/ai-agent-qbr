"""Adapters between new extraction schema and legacy metric candidates."""

from __future__ import annotations

from qbr_intelligence.metrics.models import MetricExtraction
from qbr_intelligence.pipeline.metric_scanner import (
    MetricCandidate,
    _metric_type_from_unit,
    _new_metric_id,
    build_metric_dictionary,
)


def _category_for_name(name: str | None) -> str:
    if not name:
        return ""
    for definition in build_metric_dictionary().definitions:
        if definition.name == name:
            return definition.category
    return ""


def to_metric_candidate(metric: MetricExtraction) -> MetricCandidate:
    return MetricCandidate(
        metric_id=_new_metric_id(),
        name=metric.metric_name,
        raw_value=metric.raw_value_text,
        normalized_value=metric.value,
        unit=metric.unit,
        raw_context=metric.provenance.get("snippet"),
        slide_number=metric.slide_index,
        metric_type=_metric_type_from_unit(metric.unit),
        source=metric.provenance.get("source_type", "metric_pipeline"),
        category=_category_for_name(metric.metric_name),
        extraction_confidence=metric.confidence,
        metadata={"qualifiers": metric.qualifiers, "extraction_method": metric.extraction_method},
    )
