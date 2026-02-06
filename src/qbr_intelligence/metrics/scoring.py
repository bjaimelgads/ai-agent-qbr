"""Deterministic scoring for label-value links."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from qbr_intelligence.metrics.catalog import catalog_by_id
from qbr_intelligence.metrics.models import LinkCandidate, MetricCatalogEntry, clamp_score, normalize_unit


@dataclass(frozen=True)
class ScoringWeights:
    base: float = 0.35
    unit_match: float = 0.25
    unit_mismatch: float = -0.25
    same_block: float = 0.1
    table_row: float = 0.25
    table_col: float = 0.18
    same_line: float = 0.08
    separator: float = 0.06
    proximity_scale: float = 0.18
    priority_weight: float = 0.03
    date_penalty: float = -0.2


_DATE_CONTEXT_RE = re.compile(r"\b(?:FY\d{2,4}|Q[1-4]|H[12]|20\d{2})\b", re.IGNORECASE)


def score_link(
    link: LinkCandidate,
    *,
    weights: ScoringWeights | None = None,
    catalog_map: dict[str, MetricCatalogEntry] | None = None,
) -> float:
    weights = weights or ScoringWeights()
    entry = (catalog_map or catalog_by_id()).get(link.label.metric_id)
    expected_unit = normalize_unit(entry.expected_unit if entry else "")
    unit = normalize_unit(link.value.unit)

    score = weights.base
    if expected_unit and unit:
        if expected_unit == unit:
            score += weights.unit_match
        else:
            score += weights.unit_mismatch
    if link.relation == "same_block":
        score += weights.same_block
    elif link.relation == "table_row_right":
        score += weights.table_row
    elif link.relation == "table_col_down":
        score += weights.table_col

    distance = link.features.get("distance")
    if distance is not None:
        distance_bonus = max(0.0, 1.0 - min(distance, 160) / 160)
        score += distance_bonus * weights.proximity_scale

    score += link.features.get("same_line", 0.0) * weights.same_line
    score += link.features.get("has_separator", 0.0) * weights.separator
    score += link.label.priority * weights.priority_weight

    if _DATE_CONTEXT_RE.search(link.value.context_text) and unit == "count":
        score += weights.date_penalty

    return clamp_score(score)
