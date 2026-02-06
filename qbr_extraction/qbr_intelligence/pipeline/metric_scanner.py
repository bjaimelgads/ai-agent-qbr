"""Business metric scanner for QBR extraction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import uuid
from pathlib import Path
from typing import Iterable, Sequence


_NUMBER_RE = re.compile(
    r"(?P<prefix>\$)?(?P<num>[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)"
    r"(?P<suffix>\s*[KMBkmb])?(?P<pct>%?)"
)

_TIME_RE = re.compile(
    r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>seconds|secs|s|minutes|min|hours|hrs|h)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    patterns: tuple[str, ...]
    unit_hint: str
    category: str
    slug: str | None = None
    is_calculated: bool = False
    depends_on: tuple[str, ...] = ()
    formula: str | None = None

    def regexes(self) -> list[re.Pattern[str]]:
        return [re.compile(pattern, re.IGNORECASE) for pattern in self.patterns]


@dataclass(frozen=True)
class MetricCandidate:
    metric_id: str
    name: str
    raw_value: str
    normalized_value: float | None
    unit: str | None
    raw_context: str
    slide_number: int | None
    metric_type: str
    source: str
    category: str
    extraction_confidence: float | None = None
    metadata: dict | None = None
    period_label: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    brand: str | None = None
    baseline_text: str | None = None
    baseline_type: str | None = None
    metric_catalog_id: int | None = None
    metric_catalog_slug: str | None = None
    llm_context_label: str | None = None

    def dedupe_key(self) -> tuple[str, float | None, str | None]:
        norm = None if self.normalized_value is None else round(self.normalized_value, 6)
        unit = self.unit.lower() if self.unit else None
        return (self.name.lower(), norm, unit)


@dataclass(frozen=True)
class MetricScanArtifacts:
    raw_content: str
    slides: Sequence[dict]
    metrics: Sequence[dict]
    charts: Sequence[dict]
    chunks: Sequence[dict] = field(default_factory=list)
    tables: Sequence[dict] = field(default_factory=list)
    keywords: Sequence[str] = field(default_factory=list)
    export_dir: Path | None = None

    @classmethod
    def from_export_dir(cls, export_dir: Path) -> "MetricScanArtifacts":
        raw_content = _read_text(export_dir / "02_raw_content.txt")
        slides = _read_json(export_dir / "03_slides_parsed.json")
        charts = _read_json(export_dir / "05_charts_detected.json")
        chunks = _read_json(export_dir / "08_chunks_rag.json")
        tables = _read_json(export_dir / "09_tables_extracted.json")
        keywords_payload = _read_json(export_dir / "06_keywords_topics.json")
        keywords = []
        if isinstance(keywords_payload, dict):
            keywords = keywords_payload.get("keywords", []) or []
        elif isinstance(keywords_payload, list):
            keywords = keywords_payload
        return cls(
            raw_content=raw_content,
            slides=slides,
            metrics=[],
            charts=charts,
            chunks=chunks,
            tables=tables,
            keywords=keywords,
            export_dir=export_dir,
        )


class MetricDictionary:
    """Registry for metric definitions and matching helpers."""

    def __init__(self, definitions: Sequence[MetricDefinition]) -> None:
        self._definitions = list(definitions)
        self._regex_cache: dict[str, list[re.Pattern[str]]] = {
            definition.name: definition.regexes() for definition in self._definitions
        }

    @property
    def definitions(self) -> list[MetricDefinition]:
        return list(self._definitions)

    def match_from_context(self, text: str) -> MetricDefinition | None:
        if not text:
            return None
        for definition in self._definitions:
            for regex in self._regex_cache.get(definition.name, []):
                if regex.search(text):
                    return definition
        return None

    def find_mentions(self, text: str) -> list[tuple[MetricDefinition, re.Match[str]]]:
        matches: list[tuple[MetricDefinition, re.Match[str]]] = []
        if not text:
            return matches
        for definition in self._definitions:
            for regex in self._regex_cache.get(definition.name, []):
                for match in regex.finditer(text):
                    matches.append((definition, match))
        return matches


class MetricScanner:
    """Scan extraction artifacts to produce normalized business metrics."""

    def __init__(
        self,
        dictionary: MetricDictionary,
        *,
        context_window: int = 200,
        max_context_chars: int = 800,
    ) -> None:
        self._dictionary = dictionary
        self._context_window = context_window
        self._max_context_chars = max_context_chars

    def scan(self, artifacts: MetricScanArtifacts) -> list[MetricCandidate]:
        candidates: list[MetricCandidate] = []
        candidates.extend(
            self._extract_from_text_sources(
                self._build_text_sources(artifacts),
            )
        )
        candidates.extend(self._extract_from_tables(artifacts.tables))
        return candidates

    def dedupe_exact(self, candidates: Iterable[MetricCandidate]) -> list[MetricCandidate]:
        seen: set[tuple[str, float | None, str | None]] = set()
        deduped: list[MetricCandidate] = []
        for candidate in candidates:
            key = candidate.dedupe_key()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        unique_reach_keys = {
            (m.slide_number, m.normalized_value, m.unit)
            for m in deduped
            if m.name == "Unique Reach"
        }
        if unique_reach_keys:
            deduped = [
                m
                for m in deduped
                if not (
                    m.name == "Reach"
                    and (m.slide_number, m.normalized_value, m.unit) in unique_reach_keys
                )
            ]
        return deduped

    def _extract_from_text_sources(
        self,
        sources: Sequence[dict],
    ) -> list[MetricCandidate]:
        output: list[MetricCandidate] = []
        for source in sources:
            text = source.get("text") or ""
            slide_number = source.get("slide_number")
            source_label = source.get("source", "text")
            if not text:
                continue
            for definition, match in self._dictionary.find_mentions(text):
                context = _extract_context(text, match.start(), match.end(), self._context_window)
                value_candidates = _find_value_candidates(context)
                best = _select_best_value(value_candidates, definition.unit_hint, match.span())
                if not best:
                    continue
                raw_value = best["raw_value"]
                normalized_value = best["value"]
                unit = best["unit"]
                metric_type = _metric_type_from_unit(unit)
                output.append(
                    MetricCandidate(
                        metric_id=_new_metric_id(),
                        name=definition.name,
                        raw_value=raw_value,
                        normalized_value=normalized_value,
                        unit=unit or definition.unit_hint,
                        raw_context=_truncate_context(context, self._max_context_chars),
                        slide_number=_safe_int(slide_number),
                        metric_type=metric_type,
                        source=source_label,
                        category=definition.category,
                        extraction_confidence=0.5,
                    )
                )
        return output

    def _extract_from_tables(self, tables: Sequence[dict]) -> list[MetricCandidate]:
        output: list[MetricCandidate] = []
        for table in tables:
            rows = table.get("rows") or table.get("data") or []
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, list):
                    continue
                row_text = " ".join(str(cell) for cell in row)
                for definition, match in self._dictionary.find_mentions(row_text):
                    context = _extract_context(row_text, match.start(), match.end(), self._context_window)
                    value_candidates = _find_value_candidates(context)
                    best = _select_best_value(value_candidates, definition.unit_hint, match.span())
                    if not best:
                        continue
                    output.append(
                        MetricCandidate(
                            metric_id=_new_metric_id(),
                            name=definition.name,
                            raw_value=best["raw_value"],
                            normalized_value=best["value"],
                            unit=best["unit"] or definition.unit_hint,
                            raw_context=_truncate_context(context, self._max_context_chars),
                            slide_number=_safe_int(table.get("slide_number")),
                            metric_type=_metric_type_from_unit(best["unit"]),
                            source="tables_extracted",
                            category=definition.category,
                            extraction_confidence=0.6,
                        )
                    )
        return output

    def _build_text_sources(self, artifacts: MetricScanArtifacts) -> list[dict]:
        sources: list[dict] = []
        for slide in artifacts.slides:
            sources.append(
                {
                    "text": slide.get("raw_text", ""),
                    "slide_number": slide.get("slide_number"),
                    "source": "slides_parsed",
                }
            )
        return sources


def build_metric_dictionary() -> MetricDictionary:
    definitions = [
        MetricDefinition(
            name="Click Through Rate",
            patterns=(r"\bCTR\b", r"click[- ]through rate"),
            unit_hint="percent",
            category="performance",
            slug="click_through_rate",
            is_calculated=True,
            depends_on=("Clicks", "Impressions"),
            formula="Clicks / Impressions",
        ),
        MetricDefinition(
            name="View Through Rate",
            patterns=(r"\bVTR\b", r"view[- ]through rate"),
            unit_hint="percent",
            category="performance",
            slug="view_through_rate",
            is_calculated=True,
            depends_on=("Views", "Impressions"),
            formula="Views / Impressions",
        ),
        MetricDefinition(
            name="Video Completion Rate",
            patterns=(r"\bVCR\b", r"video completion rate", r"completion rate"),
            unit_hint="percent",
            category="performance",
            slug="video_completion_rate",
            is_calculated=True,
            depends_on=("Completes", "Impressions"),
            formula="Completes / Impressions",
        ),
        MetricDefinition(
            name="Conversion Rate",
            patterns=(r"\bCVR\b", r"conversion rate"),
            unit_hint="percent",
            category="performance",
            slug="conversion_rate",
            is_calculated=True,
            depends_on=("Conversions", "Clicks"),
            formula="Conversions / Clicks",
        ),
        MetricDefinition(
            name="Cost per Acquisition",
            patterns=(r"\bCPA\b", r"cost per acquisition", r"cost per acquired user"),
            unit_hint="currency",
            category="cost",
            slug="cost_per_acquisition",
            is_calculated=True,
            depends_on=("Spend", "Acquisitions"),
            formula="Spend / Acquisitions",
        ),
        MetricDefinition(
            name="Cost per Install",
            patterns=(r"\bCPI\b", r"cost per install"),
            unit_hint="currency",
            category="cost",
            slug="cost_per_install",
            is_calculated=True,
            depends_on=("Spend", "Installs"),
            formula="Spend / Installs",
        ),
        MetricDefinition(
            name="Cost per Engagement",
            patterns=(r"\bCPE\b", r"cost per engagement"),
            unit_hint="currency",
            category="cost",
            slug="cost_per_engagement",
            is_calculated=True,
            depends_on=("Spend", "Engagements"),
            formula="Spend / Engagements",
        ),
        MetricDefinition(
            name="Cost per Click",
            patterns=(r"\bCPC\b", r"cost per click"),
            unit_hint="currency",
            category="cost",
            slug="cost_per_click",
            is_calculated=True,
            depends_on=("Spend", "Clicks"),
            formula="Spend / Clicks",
        ),
        MetricDefinition(
            name="Cost per Mille",
            patterns=(r"\bCPM\b", r"cost per mille", r"cost per thousand"),
            unit_hint="currency",
            category="cost",
            slug="cost_per_mille",
            is_calculated=True,
            depends_on=("Spend", "Impressions"),
            formula="Spend / Impressions * 1000",
        ),
        MetricDefinition(
            name="Cost per View",
            patterns=(r"\bCPV\b", r"cost per view"),
            unit_hint="currency",
            category="cost",
            slug="cost_per_view",
            is_calculated=True,
            depends_on=("Spend", "Views"),
            formula="Spend / Views",
        ),
        MetricDefinition(
            name="Install Rate",
            patterns=(r"install rate",),
            unit_hint="percent",
            category="performance",
            slug="install_rate",
            is_calculated=True,
            depends_on=("Installs", "Impressions"),
            formula="Installs / Impressions",
        ),
        MetricDefinition(
            name="Launch Rate",
            patterns=(r"launch rate",),
            unit_hint="percent",
            category="performance",
            slug="launch_rate",
            is_calculated=True,
            depends_on=("Launches", "Impressions"),
            formula="Launches / Impressions",
        ),
        MetricDefinition(
            name="Install Lift",
            patterns=(r"install lift",),
            unit_hint="percent",
            category="performance",
            slug="install_lift",
            is_calculated=True,
            depends_on=("Installs",),
            formula="(Test Installs - Control Installs) / Control Installs",
        ),
        MetricDefinition(
            name="Launch Lift",
            patterns=(r"launch lift",),
            unit_hint="percent",
            category="performance",
            slug="launch_lift",
            is_calculated=True,
            depends_on=("Launches",),
            formula="(Test Launches - Control Launches) / Control Launches",
        ),
        MetricDefinition(
            name="Return on Ad Spend",
            patterns=(r"\bROAS\b", r"return on ad spend"),
            unit_hint="count",
            category="performance",
            slug="roas",
            is_calculated=True,
            depends_on=("Revenue", "Spend"),
            formula="Revenue / Spend",
        ),
        MetricDefinition(
            name="Return on Investment",
            patterns=(r"\bROI\b", r"return on investment"),
            unit_hint="percent",
            category="performance",
            slug="roi",
            is_calculated=True,
            depends_on=("Revenue", "Spend"),
            formula="(Revenue - Spend) / Spend",
        ),
        MetricDefinition(
            name="Installs",
            patterns=(r"\binstalls?\b",),
            unit_hint="count",
            category="performance",
            slug="installs",
        ),
        MetricDefinition(
            name="Launches",
            patterns=(r"\blaunches?\b",),
            unit_hint="count",
            category="performance",
            slug="launches",
        ),
        MetricDefinition(
            name="Unique Reach",
            patterns=(r"unique reach",),
            unit_hint="count",
            category="reach",
            slug="unique_reach",
        ),
        MetricDefinition(
            name="Monthly Active Users",
            patterns=(r"\bMAU\b", r"monthly active users"),
            unit_hint="count",
            category="reach",
            slug="monthly_active_users",
        ),
        MetricDefinition(
            name="Daily Active Users",
            patterns=(r"\bDAU\b", r"daily active users"),
            unit_hint="count",
            category="reach",
            slug="daily_active_users",
        ),
        MetricDefinition(
            name="Reach",
            patterns=(r"\breach\b",),
            unit_hint="count",
            category="reach",
            slug="reach",
        ),
        MetricDefinition(
            name="Household Reach",
            patterns=(r"household reach", r"hh reach"),
            unit_hint="count",
            category="reach",
            slug="household_reach",
        ),
        MetricDefinition(
            name="Share of Voice",
            patterns=(r"share of voice", r"\bSOV\b"),
            unit_hint="percent",
            category="reach",
            slug="share_of_voice",
        ),
        MetricDefinition(
            name="GRPs",
            patterns=(r"\bGRPs?\b", r"gross rating points?"),
            unit_hint="count",
            category="reach",
            slug="grps",
        ),
        MetricDefinition(
            name="TRPs",
            patterns=(r"\bTRPs?\b", r"target rating points?"),
            unit_hint="count",
            category="reach",
            slug="trps",
        ),
        MetricDefinition(
            name="Impressions",
            patterns=(r"\bimpressions?\b",),
            unit_hint="count",
            category="reach",
            slug="impressions",
        ),
        MetricDefinition(
            name="Clicks",
            patterns=(r"\bclicks?\b",),
            unit_hint="count",
            category="performance",
            slug="clicks",
        ),
        MetricDefinition(
            name="Views",
            patterns=(r"\bviews?\b", r"video views?"),
            unit_hint="count",
            category="performance",
            slug="views",
        ),
        MetricDefinition(
            name="Completes",
            patterns=(r"\bcompletes?\b", r"video completions?"),
            unit_hint="count",
            category="performance",
            slug="completes",
        ),
        MetricDefinition(
            name="Spend",
            patterns=(r"\bspend\b",),
            unit_hint="currency",
            category="cost",
            slug="spend",
        ),
        MetricDefinition(
            name="Content Store Roadblock Spend",
            patterns=(r"\bCSRB\b", r"content store roadblock", r"content store rb"),
            unit_hint="currency",
            category="cost",
            slug="csrb_spend",
        ),
        MetricDefinition(
            name="Acquisitions",
            patterns=(r"acquired users?", r"acquisitions?"),
            unit_hint="count",
            category="performance",
            slug="acquisitions",
        ),
        MetricDefinition(
            name="Qualified Acquisitions",
            patterns=(r"qualified acquisitions?", r"qualified acquired users?"),
            unit_hint="count",
            category="performance",
            slug="qualified_acquisitions",
        ),
        MetricDefinition(
            name="Conversions",
            patterns=(r"conversions?", r"converted users?"),
            unit_hint="count",
            category="performance",
            slug="conversions",
        ),
        MetricDefinition(
            name="Purchases",
            patterns=(r"purchases?", r"transactions?"),
            unit_hint="count",
            category="performance",
            slug="purchases",
        ),
        MetricDefinition(
            name="Sign-ups",
            patterns=(r"sign[- ]?ups?", r"sign[- ]?ins?", r"sign[- ]?up"),
            unit_hint="count",
            category="performance",
            slug="sign_ups",
        ),
        MetricDefinition(
            name="Leads",
            patterns=(r"\bleads?\b",),
            unit_hint="count",
            category="performance",
            slug="leads",
        ),
        MetricDefinition(
            name="Engagements",
            patterns=(r"engagements?", r"engaged users?"),
            unit_hint="count",
            category="engagement",
            slug="engagements",
        ),
        MetricDefinition(
            name="Total Time Spent",
            patterns=(r"total time spent", r"time spent on app", r"average time spent"),
            unit_hint="time",
            category="engagement",
            slug="total_time_spent",
        ),
        MetricDefinition(
            name="Sessions",
            patterns=(r"\bsessions?\b", r"session count"),
            unit_hint="count",
            category="engagement",
            slug="sessions",
        ),
        MetricDefinition(
            name="Average Session Duration",
            patterns=(r"avg session duration", r"average session duration", r"session duration"),
            unit_hint="time",
            category="engagement",
            slug="avg_session_duration",
        ),
        MetricDefinition(
            name="Frequency",
            patterns=(r"\bfrequency\b",),
            unit_hint="count",
            category="reach",
            slug="frequency",
        ),
        MetricDefinition(
            name="Video Starts",
            patterns=(r"video starts?",),
            unit_hint="count",
            category="performance",
            slug="video_starts",
        ),
        MetricDefinition(
            name="Fill Rate",
            patterns=(r"fill rate",),
            unit_hint="percent",
            category="delivery",
            slug="fill_rate",
        ),
        MetricDefinition(
            name="Win Rate",
            patterns=(r"win rate",),
            unit_hint="percent",
            category="delivery",
            slug="win_rate",
        ),
        MetricDefinition(
            name="Viewability",
            patterns=(r"viewability",),
            unit_hint="percent",
            category="delivery",
            slug="viewability",
        ),
    ]
    return MetricDictionary(definitions)


def _find_value_candidates(text: str) -> list[dict]:
    candidates: list[dict] = []
    for match in _NUMBER_RE.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if (before.isalnum() or before == "_") and before != "$":
            continue
        if after.isalnum() or after == "_":
            continue
        raw_value = match.group(0).strip()
        value, unit = _parse_numeric_value(raw_value)
        if unit == "count":
            look = text[end : end + 10]
            if re.match(r"\s*%", look):
                unit = "percent"
                if "%" not in raw_value:
                    raw_value = f"{raw_value}%"
        if value is None:
            continue
        candidates.append(
            {
                "raw_value": raw_value,
                "value": value,
                "unit": unit,
                "span": (start, end),
            }
        )
    for match in _TIME_RE.finditer(text):
        raw_value = match.group(0).strip()
        value = float(match.group("num"))
        unit = match.group("unit").lower()
        candidates.append(
            {
                "raw_value": raw_value,
                "value": value,
                "unit": "time",
                "span": match.span(),
            }
        )
    return candidates


def _select_best_value(
    candidates: list[dict],
    unit_hint: str,
    match_span: tuple[int, int],
    *,
    max_distance: int = 120,
) -> dict | None:
    if not candidates:
        return None
    unit_hint = (unit_hint or "").lower()
    if unit_hint == "percent":
        candidates = [c for c in candidates if c.get("unit") == "percent"]
    elif unit_hint == "currency":
        candidates = [c for c in candidates if c.get("unit") == "currency"]
    elif unit_hint == "time":
        candidates = [c for c in candidates if c.get("unit") == "time"]
    elif unit_hint == "count":
        candidates = [c for c in candidates if c.get("unit") in {None, "count"}]
        with_suffix = [
            c
            for c in candidates
            if re.search(r"[kmb]", str(c.get("raw_value", "")), re.IGNORECASE)
        ]
        if with_suffix:
            candidates = with_suffix
    if not candidates:
        return None

    match_start, match_end = match_span
    best = None
    best_distance = None
    for cand in candidates:
        c_start, c_end = cand.get("span", (0, 0))
        if c_end <= match_start:
            distance = match_start - c_end
        elif c_start >= match_end:
            distance = c_start - match_end
        else:
            distance = 0
        if best_distance is None or distance < best_distance:
            best = cand
            best_distance = distance
    if best_distance is None or best_distance > max_distance:
        return None
    return best


def _parse_numeric_value(raw_value: str) -> tuple[float | None, str | None]:
    text = raw_value.replace(",", "").strip()
    unit = None
    if "%" in text:
        unit = "percent"
    if "$" in text:
        unit = "currency"
    match = _NUMBER_RE.search(text)
    if not match:
        return None, unit
    number = match.group("num")
    try:
        value = float(number)
    except ValueError:
        return None, unit
    suffix = (match.group("suffix") or "").strip().lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    elif suffix == "b":
        value *= 1_000_000_000
    if unit is None:
        unit = "count"
    return value, unit


def _unit_matches_hint(unit: str | None, hint: str) -> bool:
    hint = (hint or "").lower()
    unit = (unit or "").lower()
    if hint == "percent":
        return unit == "percent"
    if hint == "currency":
        return unit == "currency"
    if hint == "time":
        return unit == "time"
    if hint == "count":
        return unit in {"count", ""}
    return True


def _metric_type_from_unit(unit: str | None) -> str:
    if unit == "percent":
        return "percentage"
    if unit == "currency":
        return "currency"
    if unit == "time":
        return "time"
    return "count"


def _extract_context(text: str, start: int, end: int, window: int) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    return text[left:right].strip()


def _truncate_context(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _read_json(path: Path) -> list | dict:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _new_metric_id() -> str:
    return uuid.uuid4().hex
