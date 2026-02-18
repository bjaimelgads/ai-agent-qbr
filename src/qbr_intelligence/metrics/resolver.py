"""Resolve ambiguities and dedupe extracted metrics."""

from __future__ import annotations

from collections import defaultdict
import re

from qbr_intelligence.metrics.catalog import catalog_by_id
from qbr_intelligence.metrics.models import MetricCatalogEntry
from qbr_intelligence.metrics.models import MetricExtraction


_SOURCE_PRIORITY = {
    "table_cell": 3,
    "slide_text": 2,
    "speaker_notes": 1,
}


def resolve_metrics(
    metrics: list[MetricExtraction],
    *,
    catalog_map: dict[str, MetricCatalogEntry] | None = None,
) -> list[MetricExtraction]:
    metrics = [_resolve_reach(metric, catalog_map=catalog_map) for metric in metrics]
    metrics = _dedupe_by_preference(metrics)
    metrics = _drop_redundant_reach(metrics)
    return metrics


def _resolve_reach(
    metric: MetricExtraction,
    *,
    catalog_map: dict[str, MetricCatalogEntry] | None = None,
) -> MetricExtraction:
    if metric.metric_id != "reach":
        return metric
    snippet = (metric.label_text + " " + metric.provenance.get("snippet", "")).lower()
    unique_tokens = (catalog_map or catalog_by_id()).get("unique_reach")
    if unique_tokens:
        for token in unique_tokens.disambiguation:
            if token.lower() in snippet:
                return MetricExtraction(
                    deck_id=metric.deck_id,
                    slide_index=metric.slide_index,
                    metric_id="unique_reach",
                    metric_name="Unique Reach",
                    value=metric.value,
                    unit=metric.unit,
                    scale=metric.scale,
                    raw_value_text=metric.raw_value_text,
                    label_text=metric.label_text,
                    qualifiers=metric.qualifiers,
                    confidence=metric.confidence,
                    extraction_method=metric.extraction_method,
                    provenance=metric.provenance,
                )
    return metric


def _drop_redundant_reach(metrics: list[MetricExtraction]) -> list[MetricExtraction]:
    unique_keys = {
        (m.slide_index, m.value, m.unit)
        for m in metrics
        if m.metric_id == "unique_reach"
    }
    if not unique_keys:
        return metrics
    output = []
    for metric in metrics:
        if metric.metric_id == "reach" and (metric.slide_index, metric.value, metric.unit) in unique_keys:
            continue
        output.append(metric)
    return output


def _dedupe_by_preference(metrics: list[MetricExtraction]) -> list[MetricExtraction]:
    grouped: dict[tuple, list[MetricExtraction]] = defaultdict(list)
    for metric in metrics:
        key = (
            metric.slide_index,
            metric.metric_id,
            metric.value,
            metric.unit,
            tuple(sorted(metric.qualifiers.items())),
        )
        grouped[key].append(metric)

    output: list[MetricExtraction] = []
    for items in grouped.values():
        if len(items) == 1:
            output.append(items[0])
            continue
        items_sorted = sorted(
            items,
            key=lambda m: (
                m.confidence,
                _SOURCE_PRIORITY.get(m.provenance.get("source_type", ""), 0),
            ),
            reverse=True,
        )
        output.append(items_sorted[0])
    return output
