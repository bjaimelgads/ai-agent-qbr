"""Shared intent-based filtering helpers for retrieval and metric scoping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import re
from typing import Any

from qbr_intelligence.metric_qa import MetricQueryEngine
from qbr_intelligence.metric_qa.intent import DeterministicIntentExtractor
from qbr_intelligence.schemas.metric_qa import QueryIntent

_REGION_ALIAS = {
    "US": {"US", "USA", "UNITED STATES"},
    "EMEA": {"EMEA"},
    "GLOBAL": {"GLOBAL"},
}

_PERIOD_HALF_RE = re.compile(r"\b(H[12])\s*(?:FY)?\s*(20\d{2}|\d{2})\b", re.IGNORECASE)
_PERIOD_FY_HALF_RE = re.compile(r"\bFY\s*(20\d{2}|\d{2})\s*(H[12])\b", re.IGNORECASE)
_PERIOD_QUARTER_RE = re.compile(r"\b(Q[1-4])\s*(?:FY)?\s*(20\d{2}|\d{2})\b", re.IGNORECASE)


@dataclass(frozen=True)
class DocumentFilters:
    clients: set[str]
    regions: set[str]
    periods: set[str]

    def enabled(self) -> bool:
        return bool(self.clients or self.regions or self.periods)


def normalize_token(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
    return re.sub(r"\s+", " ", normalized)


def _normalize_year(raw: str) -> str:
    if len(raw) == 2:
        return f"20{raw}"
    return raw


def period_tokens_from_text(text: str | None) -> set[str]:
    if not text:
        return set()
    source = normalize_token(text)
    tokens: set[str] = set()
    for half, year in _PERIOD_HALF_RE.findall(source):
        year4 = _normalize_year(year)
        tokens.add(f"{half.upper()} {year4}")
        tokens.add(f"FY{year4[2:]} {half.upper()}")
    for year, half in _PERIOD_FY_HALF_RE.findall(source):
        year4 = _normalize_year(year)
        tokens.add(f"{half.upper()} {year4}")
        tokens.add(f"FY{year4[2:]} {half.upper()}")
    for quarter, year in _PERIOD_QUARTER_RE.findall(source):
        year4 = _normalize_year(year)
        tokens.add(f"{quarter.upper()} {year4}")
        tokens.add(f"FY{year4[2:]} {quarter.upper()}")
    return tokens


def _region_tokens_from_text(text: str | None) -> set[str]:
    if not text:
        return set()
    source = normalize_token(text)
    found: set[str] = set()
    for canonical, aliases in _REGION_ALIAS.items():
        if any(alias in source for alias in aliases):
            found.add(canonical)
    return found


def _client_tokens_from_document(document: Any) -> set[str]:
    tokens: set[str] = set()
    if getattr(document, "client_name", None):
        tokens.add(normalize_token(str(document.client_name)))
    filename = getattr(document, "filename", None)
    if filename:
        normalized = normalize_token(str(filename))
        if "DISNEY" in normalized:
            tokens.add("DISNEY")
            tokens.add("DISNEY+")
        if "NETFLIX" in normalized:
            tokens.add("NETFLIX")
        if "HULU" in normalized:
            tokens.add("HULU")
    return tokens


def _period_tokens_from_document(document: Any) -> set[str]:
    tokens = period_tokens_from_text(getattr(document, "period", None))
    filename = getattr(document, "filename", None)
    if filename:
        tokens.update(period_tokens_from_text(str(filename)))
    return tokens


def _region_tokens_from_document(document: Any) -> set[str]:
    joined = " ".join(
        str(part)
        for part in (
            getattr(document, "filename", None),
            getattr(document, "file_path", None),
            getattr(document, "period", None),
        )
        if part
    )
    return _region_tokens_from_text(joined)


def build_document_filters_from_intent(intent: QueryIntent) -> DocumentFilters:
    clients: set[str] = set()
    for client in intent.client or []:
        if client:
            clients.add(normalize_token(client))

    regions: set[str] = set()
    for region in intent.region or []:
        if not region:
            continue
        normalized = normalize_token(region)
        if normalized in _REGION_ALIAS:
            regions.add(normalized)
            continue
        for canonical, aliases in _REGION_ALIAS.items():
            if normalized in aliases:
                regions.add(canonical)
                break

    periods: set[str] = set()
    for period in intent.period or []:
        value = getattr(period, "value", None)
        if isinstance(value, str):
            periods.update(period_tokens_from_text(value))
        start = getattr(period, "start", None)
        end = getattr(period, "end", None)
        kind = getattr(period, "type", None)
        if isinstance(start, date) and isinstance(end, date):
            year = end.year if end.month >= 7 else start.year
            if isinstance(kind, str) and kind.lower() == "half":
                half = "H1" if start.month <= 3 else "H2"
                periods.add(f"{half} {year}")
                periods.add(f"FY{str(year)[2:]} {half}")
    return DocumentFilters(clients=clients, regions=regions, periods=periods)


async def build_document_filters_from_query(
    *,
    query: str,
    engine: MetricQueryEngine,
) -> tuple[DocumentFilters, QueryIntent]:
    await engine._load_catalogs()
    anchor_date = await engine._select_anchor_date(query)
    extractor = DeterministicIntentExtractor(engine._catalog or [], engine._clients or [])
    intent, _ = extractor.extract(query, anchor_date=anchor_date)
    return build_document_filters_from_intent(intent), intent


def score_document_match(document: Any, filters: DocumentFilters) -> tuple[bool, float]:
    boost = 0.0

    if filters.clients:
        doc_clients = _client_tokens_from_document(document)
        if doc_clients:
            if doc_clients & filters.clients:
                boost += 0.25
            else:
                return False, 0.0

    if filters.regions:
        doc_regions = _region_tokens_from_document(document)
        if doc_regions:
            if doc_regions & filters.regions:
                boost += 0.20
            else:
                return False, 0.0

    if filters.periods:
        doc_periods = _period_tokens_from_document(document)
        if doc_periods:
            if doc_periods & filters.periods:
                boost += 0.20
            else:
                return False, 0.0

    return True, boost
