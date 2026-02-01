"""Heuristic router for metric queries."""

from __future__ import annotations

import re
from qbr_intelligence.metrics.catalog import build_metric_catalog


_METRIC_KEYWORDS = {
    "cpa",
    "cpi",
    "ctr",
    "cpc",
    "cpm",
    "cpv",
    "reach",
    "impressions",
    "installs",
    "conversions",
    "roas",
    "roi",
}


class MetricQueryRouter:
    def __init__(self) -> None:
        catalog = build_metric_catalog()
        aliases = []
        for entry in catalog:
            for alias in entry.aliases:
                if alias.alias:
                    aliases.append(alias.alias)
        escaped = [re.escape(alias) for alias in aliases if alias]
        self._pattern = re.compile(r"\b(" + "|".join(sorted(set(escaped), key=len, reverse=True)) + r")\b", re.IGNORECASE)

    def is_metric_query(self, query: str) -> bool:
        if not query:
            return False
        lowered = query.lower()
        if self._pattern.search(lowered):
            return True
        return any(keyword in lowered for keyword in _METRIC_KEYWORDS)
