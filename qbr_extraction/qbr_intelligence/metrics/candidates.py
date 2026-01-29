"""Candidate generation for labels and values."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from qbr_intelligence.metrics.catalog import build_metric_catalog
from qbr_intelligence.metrics.models import (
    CandidateBundle,
    LabelCandidate,
    MetricCatalogEntry,
    ValueCandidate,
    UNIT_COUNT,
    UNIT_CURRENCY,
    UNIT_PERCENT,
    UNIT_RATIO,
    UNIT_UNKNOWN,
    SCALE_B,
    SCALE_K,
    SCALE_M,
    SCALE_ONES,
)


_RANGE_RE = re.compile(
    r"(?<![\w.])(?P<start>[+-]?\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*"
    r"(?P<end>[+-]?\d+(?:\.\d+)?)(?P<suffix>%?)"
    r"(?!\w)",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<prefix>\$)?(?P<num>[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)"
    r"(?P<suffix>\s*[KMBkmb])?(?P<pct>%?)(?P<plus>\+?)"
    r"(?!\w)",
)

_RATIO_RE = re.compile(r"(?<![\w.])(?P<num>\d+(?:\.\d+)?)\s*(?:x|:1|/1)(?!\w)", re.IGNORECASE)

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass(frozen=True)
class AliasPattern:
    metric_id: str
    metric_name: str
    alias: str
    regex: re.Pattern[str]
    priority: int


def _compile_alias(alias: str) -> re.Pattern[str]:
    parts = [part for part in re.split(r"[\s\-]+", alias.strip()) if part]
    escaped = r"[\s\-]+".join(re.escape(part) for part in parts)
    pattern = rf"\b{escaped}\b"
    return re.compile(pattern, re.IGNORECASE)


def build_alias_patterns(entries: list[MetricCatalogEntry]) -> list[AliasPattern]:
    patterns: list[AliasPattern] = []
    for entry in entries:
        for alias_spec in entry.aliases:
            regex = _compile_alias(alias_spec.alias)
            if alias_spec.pattern:
                regex = re.compile(alias_spec.pattern, re.IGNORECASE)
            priority = alias_spec.priority if alias_spec.priority is not None else entry.priority
            patterns.append(
                AliasPattern(
                    metric_id=entry.metric_id,
                    metric_name=entry.name,
                    alias=alias_spec.alias,
                    regex=regex,
                    priority=priority,
                )
            )
    return patterns


_ALIAS_CACHE: list[AliasPattern] | None = None


def _alias_patterns() -> list[AliasPattern]:
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        _ALIAS_CACHE = build_alias_patterns(build_metric_catalog())
    return _ALIAS_CACHE


def _is_noise_number(raw_value: str, context: str) -> bool:
    text = raw_value.strip()
    has_unit = any(ch in text for ch in ["%", "$", "K", "M", "B", "k", "m", "b"])
    if _YEAR_RE.fullmatch(text) and not any(ch in text for ch in ["%", "$", "K", "M", "B", "k", "m", "b"]):
        return True
    context_lower = context.lower()
    if "slide" in context_lower or "page" in context_lower:
        return True
    if re.search(r"\bFY\d{2,4}\b", context, re.IGNORECASE) and not has_unit and _YEAR_RE.fullmatch(text):
        return True
    if re.search(r"\bQ[1-4]\b", context, re.IGNORECASE) and _YEAR_RE.fullmatch(text):
        return True
    if re.search(r"\b(19|20)\d{2}\b", context) and len(text) == 4 and not has_unit:
        return True
    if re.search(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b", context, re.IGNORECASE) and not has_unit:
        return True
    if re.search(r"\b(?:FY|Q\d|H\d)\b", context, re.IGNORECASE) and len(text) <= 2:
        return True
    return False


def _parse_numeric(raw_value: str) -> tuple[float | None, str, str]:
    text = raw_value.replace(",", "").strip()
    unit = UNIT_UNKNOWN
    scale = SCALE_ONES
    if "%" in text:
        unit = UNIT_PERCENT
    if "$" in text:
        unit = UNIT_CURRENCY
    match = _NUMBER_RE.search(text)
    if not match:
        return None, unit, scale
    try:
        value = float(match.group("num"))
    except ValueError:
        return None, unit, scale
    suffix = (match.group("suffix") or "").strip().lower()
    if suffix == "k":
        value *= 1_000
        scale = SCALE_K
    elif suffix == "m":
        value *= 1_000_000
        scale = SCALE_M
    elif suffix == "b":
        value *= 1_000_000_000
        scale = SCALE_B
    if unit == UNIT_UNKNOWN:
        unit = UNIT_COUNT
    return value, unit, scale


def _extract_range_candidates(text: str) -> list[tuple[str, float, str, str]]:
    candidates: list[tuple[str, float, str, str]] = []
    for match in _RANGE_RE.finditer(text):
        start = float(match.group("start"))
        end = float(match.group("end"))
        raw = match.group(0).strip()
        avg = (start + end) / 2.0
        unit = UNIT_PERCENT if match.group("suffix") == "%" else UNIT_COUNT
        candidates.append((raw, avg, unit, SCALE_ONES))
    return candidates


def _extract_ratio_candidates(text: str) -> list[tuple[str, float, str, str]]:
    candidates: list[tuple[str, float, str, str]] = []
    for match in _RATIO_RE.finditer(text):
        raw = match.group(0).strip()
        value = float(match.group("num"))
        candidates.append((raw, value, UNIT_RATIO, SCALE_ONES))
    return candidates


def extract_value_candidates(text: str) -> list[ValueCandidate]:
    candidates: list[ValueCandidate] = []
    for raw, value, unit, scale in _extract_range_candidates(text):
        span = _safe_span(text, raw)
        context = _context_window(text, span)
        if _is_noise_number(raw, context):
            continue
        qualifiers = extract_qualifiers(context)
        candidates.append(
            ValueCandidate(
                raw_value_text=raw,
                normalized_value=value,
                unit=unit,
                scale=scale,
                slide_index=-1,
                source_type="",
                block_id="",
                span=span,
                context_text=context,
                qualifiers=qualifiers | {"range": raw},
                is_range=True,
            )
        )
    for raw, value, unit, scale in _extract_ratio_candidates(text):
        span = _safe_span(text, raw)
        context = _context_window(text, span)
        if _is_noise_number(raw, context):
            continue
        qualifiers = extract_qualifiers(context)
        candidates.append(
            ValueCandidate(
                raw_value_text=raw,
                normalized_value=value,
                unit=unit,
                scale=scale,
                slide_index=-1,
                source_type="",
                block_id="",
                span=span,
                context_text=context,
                qualifiers=qualifiers,
                is_range=False,
            )
        )
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0).strip()
        span = match.span()
        context = _context_window(text, span)
        if _is_noise_number(raw, context):
            continue
        value, unit, scale = _parse_numeric(raw)
        if value is None:
            continue
        qualifiers = extract_qualifiers(context)
        candidates.append(
            ValueCandidate(
                raw_value_text=raw,
                normalized_value=value,
                unit=unit,
                scale=scale,
                slide_index=-1,
                source_type="",
                block_id="",
                span=span,
                context_text=context,
                qualifiers=qualifiers,
                is_range=False,
            )
        )
    return candidates


def extract_label_candidates(
    text: str,
    source_type: str,
    block_id: str,
    slide_index: int,
    *,
    alias_patterns: list[AliasPattern] | None = None,
) -> list[LabelCandidate]:
    labels: list[LabelCandidate] = []
    patterns = alias_patterns or _alias_patterns()
    for alias in patterns:
        for match in alias.regex.finditer(text):
            labels.append(
                LabelCandidate(
                    metric_id=alias.metric_id,
                    metric_name=alias.metric_name,
                    alias=alias.alias,
                    label_text=match.group(0),
                    slide_index=slide_index,
                    source_type=source_type,
                    block_id=block_id,
                    span=match.span(),
                    context_text=text,
                    priority=alias.priority,
                )
            )
    return labels


def extract_qualifiers(text: str) -> dict[str, str]:
    qualifiers: dict[str, str] = {}
    if not text:
        return qualifiers
    if re.search(r"\bUS\b|\bUnited States\b|\bDomestic\b", text, re.IGNORECASE):
        qualifiers["geo"] = "US"
    elif re.search(r"\bGlobal\b|\bWorldwide\b", text, re.IGNORECASE):
        qualifiers["geo"] = "Global"
    elif re.search(r"\bEMEA\b", text, re.IGNORECASE):
        qualifiers["geo"] = "EMEA"
    elif re.search(r"\bAPAC\b", text, re.IGNORECASE):
        qualifiers["geo"] = "APAC"
    elif re.search(r"\bLATAM\b", text, re.IGNORECASE):
        qualifiers["geo"] = "LATAM"
    if re.search(r"\biOS\b", text, re.IGNORECASE):
        qualifiers["platform"] = "iOS"
    elif re.search(r"\bAndroid\b", text, re.IGNORECASE):
        qualifiers["platform"] = "Android"
    elif re.search(r"\bwebOS\b", text, re.IGNORECASE):
        qualifiers["platform"] = "webOS"
    if re.search(r"\bnew\b.*\busers\b|\bnew users\b", text, re.IGNORECASE):
        qualifiers["audience"] = "new_users"
    if re.search(r"\blapsed\b", text, re.IGNORECASE):
        qualifiers["audience"] = "lapsed"
    if re.search(r"\bFY\s*\d{2,4}\b", text, re.IGNORECASE):
        qualifiers["timeframe"] = "FY"
    if re.search(r"\bQ[1-4]\b", text, re.IGNORECASE):
        qualifiers["timeframe"] = "Q"
    if re.search(r"\bH[12]\b", text, re.IGNORECASE):
        qualifiers["timeframe"] = "H"
    return qualifiers


def build_candidates(
    *,
    slide_index: int,
    source_type: str,
    block_id: str,
    text: str,
    alias_patterns: list[AliasPattern] | None = None,
) -> CandidateBundle:
    labels = extract_label_candidates(
        text,
        source_type,
        block_id,
        slide_index,
        alias_patterns=alias_patterns,
    )
    values = []
    for candidate in extract_value_candidates(text):
        values.append(
            ValueCandidate(
                raw_value_text=candidate.raw_value_text,
                normalized_value=candidate.normalized_value,
                unit=candidate.unit,
                scale=candidate.scale,
                slide_index=slide_index,
                source_type=source_type,
                block_id=block_id,
                span=candidate.span,
                context_text=candidate.context_text,
                qualifiers=candidate.qualifiers,
                is_range=candidate.is_range,
            )
        )
    return CandidateBundle(labels=tuple(labels), values=tuple(values))


def _context_window(text: str, span: tuple[int, int], window: int = 80) -> str:
    start = max(0, span[0] - window)
    end = min(len(text), span[1] + window)
    return text[start:end].strip()


def _safe_span(text: str, token: str) -> tuple[int, int]:
    try:
        start = text.index(token)
        return (start, start + len(token))
    except ValueError:
        return (0, len(token))
