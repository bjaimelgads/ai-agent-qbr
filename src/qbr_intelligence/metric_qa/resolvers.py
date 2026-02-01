"""Deterministic entity resolution for metric QA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import os
import re
from typing import Iterable

from qbr_intelligence.metrics.models import MetricCatalogEntry
from qbr_intelligence.metrics.regions import REGION_EMEA, REGION_US, infer_region_from_text

from .utils import (
    DateRange,
    normalize_text,
    parse_year,
    quarter_bounds,
    half_bounds,
    year_bounds,
)


@dataclass(frozen=True)
class Resolution:
    value: str | None
    confidence: float
    candidates: list[str]
    reason: str | None = None


def _fiscal_start_month() -> int:
    raw = (os.getenv("FISCAL_YEAR_START_MONTH") or "10").strip()
    try:
        value = int(raw)
    except ValueError:
        return 10
    return value if 1 <= value <= 12 else 10


class MetricResolver:
    def __init__(self, catalog: Iterable[MetricCatalogEntry]):
        self._catalog = list(catalog)
        self._patterns = self._build_patterns(self._catalog)

    @staticmethod
    def _build_patterns(catalog: list[MetricCatalogEntry]) -> list[tuple[MetricCatalogEntry, re.Pattern[str], float]]:
        patterns: list[tuple[MetricCatalogEntry, re.Pattern[str], float]] = []
        for entry in catalog:
            for alias in entry.aliases:
                if alias.pattern:
                    try:
                        pattern = re.compile(alias.pattern, re.IGNORECASE)
                        patterns.append((entry, pattern, float(alias.priority or 0)))
                    except re.error:
                        continue
                if alias.alias:
                    escaped = re.escape(alias.alias)
                    pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
                    patterns.append((entry, pattern, float(alias.priority or 0)))
        return patterns

    def resolve(self, query: str) -> tuple[list[str], dict[str, float]]:
        if not query:
            return [], {}
        text = normalize_text(query)
        scores: dict[str, float] = {}
        for entry, pattern, alias_priority in self._patterns:
            if not pattern.search(text):
                continue
            score = entry.priority + alias_priority
            if entry.disambiguation:
                if any(token in text for token in entry.disambiguation):
                    score += 0.6
                else:
                    score -= 0.4
            scores[entry.metric_id] = max(scores.get(entry.metric_id, -1.0), score)
        ordered = [k for k, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        return ordered, scores


class ClientResolver:
    def __init__(self, clients: Iterable[str]):
        self._clients = [client for client in clients if client]

    def resolve(self, query: str) -> Resolution:
        if not query or not self._clients:
            return Resolution(value=None, confidence=0.0, candidates=[])
        lowered = normalize_text(query)
        matches: list[tuple[str, int]] = []
        for client in self._clients:
            name = normalize_text(client)
            if not name:
                continue
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                matches.append((client, len(name)))
        if not matches:
            return Resolution(value=None, confidence=0.0, candidates=[])
        matches.sort(key=lambda item: item[1], reverse=True)
        best, best_len = matches[0]
        tied = [client for client, length in matches if length == best_len]
        if len(tied) > 1:
            return Resolution(value=None, confidence=0.3, candidates=tied, reason="multiple_client_matches")
        return Resolution(value=best, confidence=0.8, candidates=[best])


class RegionResolver:
    _ALIASES = {
        "us": REGION_US,
        "u.s.": REGION_US,
        "usa": REGION_US,
        "united states": REGION_US,
        "domestic": REGION_US,
        "emea": REGION_EMEA,
        "europe": REGION_EMEA,
        "global": "GLOBAL",
        "worldwide": "GLOBAL",
    }

    def resolve(self, query: str) -> Resolution:
        if not query:
            return Resolution(value=None, confidence=0.0, candidates=[])
        lowered = normalize_text(query)
        matches = []
        for alias, value in self._ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                matches.append(value)
        inferred = infer_region_from_text(query)
        if inferred:
            matches.append(inferred)
        if not matches:
            return Resolution(value=None, confidence=0.0, candidates=[])
        unique = list(dict.fromkeys(matches))
        if len(unique) > 1:
            return Resolution(value=None, confidence=0.3, candidates=unique, reason="multiple_regions")
        return Resolution(value=unique[0], confidence=0.9, candidates=unique)


class PeriodResolver:
    _Q_PATTERN = re.compile(r"\bQ(?P<q>[1-4])\s*(?:FY)?\s*'?(?P<y>\d{2,4})\b", re.IGNORECASE)
    _Q_PATTERN_REV = re.compile(r"\b(?P<q>[1-4])Q\s*'?(?P<y>\d{2,4})\b", re.IGNORECASE)
    _H_PATTERN = re.compile(r"\bH(?P<h>[12])\s*(?:FY)?\s*'?(?P<y>\d{2,4})\b", re.IGNORECASE)
    _H_PATTERN_REV = re.compile(r"\bFY\s*'?(?P<y>\d{2,4})\s*H(?P<h>[12])\b", re.IGNORECASE)
    _FY_PATTERN = re.compile(r"\bFY\s*'?(?P<y>\d{2,4})\b", re.IGNORECASE)
    _RELATIVE_PATTERN = re.compile(
        r"\b(last|past|previous)\s+(?P<num>\d+\s+)?(?P<unit>quarters?|years?|halves|half)\b",
        re.IGNORECASE,
    )
    _Q_RANGE_PATTERN = re.compile(
        r"\bQ(?P<q1>[1-4])\s*(?:vs|and|to|-)\s*Q(?P<q2>[1-4])\s*(?P<y>\d{2,4})\b",
        re.IGNORECASE,
    )

    def resolve(self, query: str, *, anchor_date: date | None) -> tuple[DateRange | None, str | None, str | None]:
        if not query:
            return None, None, None
        text = normalize_text(query)
        fiscal_start = _fiscal_start_month()

        range_match = self._Q_RANGE_PATTERN.search(text)
        if range_match:
            year = parse_year(range_match.group("y"))
            q1 = int(range_match.group("q1"))
            q2 = int(range_match.group("q2"))
            if year:
                ranges = [
                    quarter_bounds(year, q1, fiscal_start),
                    quarter_bounds(year, q2, fiscal_start),
                ]
                starts = [rng.start for rng in ranges if rng.start]
                ends = [rng.end for rng in ranges if rng.end]
                if starts and ends:
                    label = f"Q{min(q1, q2)}-Q{max(q1, q2)} {year}"
                    return DateRange(start=min(starts), end=max(ends)), label, "range"

        multi_quarters = self._extract_quarters(text)
        if len(multi_quarters) >= 2:
            ranges = [quarter_bounds(year, quarter, fiscal_start) for year, quarter in multi_quarters]
            starts = [rng.start for rng in ranges if rng.start]
            ends = [rng.end for rng in ranges if rng.end]
            if starts and ends:
                label = f"Q{multi_quarters[0][1]}-Q{multi_quarters[-1][1]} {multi_quarters[0][0]}"
                return DateRange(start=min(starts), end=max(ends)), label, "range"

        multi_halves = self._extract_halves(text)
        if len(multi_halves) >= 2:
            ranges = [half_bounds(year, half, fiscal_start) for year, half in multi_halves]
            starts = [rng.start for rng in ranges if rng.start]
            ends = [rng.end for rng in ranges if rng.end]
            if starts and ends:
                label = f"H{multi_halves[0][1]}-H{multi_halves[-1][1]} {multi_halves[0][0]}"
                return DateRange(start=min(starts), end=max(ends)), label, "range"

        match = self._Q_PATTERN.search(text) or self._Q_PATTERN_REV.search(text)
        if match:
            year = parse_year(match.group("y"))
            quarter = int(match.group("q"))
            if year:
                bounds = quarter_bounds(year, quarter, fiscal_start)
                return bounds, f"Q{quarter} {year}", "quarter"

        match = self._H_PATTERN.search(text) or self._H_PATTERN_REV.search(text)
        if match:
            year = parse_year(match.group("y"))
            half = int(match.group("h"))
            if year:
                bounds = half_bounds(year, half, fiscal_start)
                return bounds, f"H{half} {year}", "half"

        match = self._FY_PATTERN.search(text)
        if match:
            year = parse_year(match.group("y"))
            if year:
                bounds = year_bounds(year, fiscal_start)
                return bounds, f"FY{str(year)[-2:]}", "year"

        rel = self._RELATIVE_PATTERN.search(text)
        if rel:
            unit = rel.group("unit").lower()
            num_raw = rel.group("num")
            count = int(num_raw.strip()) if num_raw else 1
            if anchor_date is None:
                anchor_date = date.today()
            if unit.startswith("quarter"):
                months = 3 * count
            elif unit.startswith("half"):
                months = 6 * count
            else:
                months = 12 * count
            end = anchor_date
            start = anchor_date - timedelta(days=30 * months)
            return DateRange(start=start, end=end), f"last {count} {unit}", "relative"

        if "last quarter" in text or "previous quarter" in text:
            if anchor_date is None:
                anchor_date = date.today()
            end = anchor_date
            start = anchor_date - timedelta(days=90)
            return DateRange(start=start, end=end), "last quarter", "relative"

        if "last year" in text or "previous year" in text:
            if anchor_date is None:
                anchor_date = date.today()
            end = anchor_date
            start = anchor_date - timedelta(days=365)
            return DateRange(start=start, end=end), "last year", "relative"

        return None, None, None

    def _extract_quarters(self, text: str) -> list[tuple[int, int]]:
        matches: list[tuple[int, int]] = []
        for match in self._Q_PATTERN.finditer(text):
            year = parse_year(match.group("y"))
            quarter = int(match.group("q"))
            if year:
                matches.append((year, quarter))
        for match in self._Q_PATTERN_REV.finditer(text):
            year = parse_year(match.group("y"))
            quarter = int(match.group("q"))
            if year:
                matches.append((year, quarter))
        return sorted(set(matches))

    def _extract_halves(self, text: str) -> list[tuple[int, int]]:
        matches: list[tuple[int, int]] = []
        for match in self._H_PATTERN.finditer(text):
            year = parse_year(match.group("y"))
            half = int(match.group("h"))
            if year:
                matches.append((year, half))
        for match in self._H_PATTERN_REV.finditer(text):
            year = parse_year(match.group("y"))
            half = int(match.group("h"))
            if year:
                matches.append((year, half))
        return sorted(set(matches))
