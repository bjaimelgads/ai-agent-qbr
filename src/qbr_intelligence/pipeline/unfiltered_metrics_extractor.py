"""Deterministic unfiltered metric extraction from 02b_raw_content_by_box.txt.

Designed for high precision with explicit confidence scoring and review gating.
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Sequence

from qbr_intelligence.pipeline.table_ids import compute_table_uuid, normalize_table_rows
from sqlalchemy import create_engine, text

PAGE_SPLIT_RE = re.compile(r"<!-- PAGE (\d+) -->")
DELIM_LINE = "- - - -"
OVERALL_SLIDE_RE = re.compile(
    r"\b(?:at\s+a\s+glance|overall\s+performance|overall\s+summary|executive\s+summary|summary)\b",
    re.IGNORECASE,
)

VALUE_LINE_RE = re.compile(
    r"^\s*(?:\$)?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*[KMBkmb])?(?:%|x|"
    r"\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))?\+?\s*$",
    re.IGNORECASE,
)
PRIMARY_VALUE_RE = re.compile(
    r"(?<![A-Za-z])(?:\$)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*[KMBkmb])?(?!\d)(?:%|x|"
    r"\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))?\+?",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(
    r"(?P<prefix>\$)?(?P<num>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?P<suffix>\s*[KMBkmb])?(?P<pct>%?)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
TIME_RE = re.compile(
    r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>seconds|secs|s|minutes|mins|min|hours|hrs|h)\b",
    re.IGNORECASE,
)
RATIO_RE = re.compile(
    r"(?P<num>[+-]?\d+(?:\.\d+)?)\s*x\b",
    re.IGNORECASE,
)
METRIC_HINT_RE = re.compile(
    r"\b(?:"
    r"ctr|vtr|vcr|cvr|roas|roi|cpa|cpi|cpe|cpc|cpm|cpv|"
    r"rate|lift|reach|impression|click|view|complete|spend|acquisition|conversion|purchase|"
    r"engagement|session|duration|frequency|install|launch|mau|dau|sov|grps?|trps?|investment|"
    r"time spent|open rate|streaming hours|video starts?|viewability|fill rate|win rate"
    r")\b",
    re.IGNORECASE,
)
FISCAL_LABEL_RE = re.compile(
    r"^(?:fy\s*\d{2,4}|q\s*[1-4]|h\s*[12])$",
    re.IGNORECASE,
)

COUNTRY_LABELS = {
    "united kingdom",
    "uk",
    "spain",
    "germany",
    "italy",
    "france",
    "poland",
    "netherlands",
    "hungary",
    "sweden",
    "denmark",
    "finland",
    "norway",
    "greece",
    "turkey",
    "czech republic",
    "czechia",
    "romania",
    "austria",
    "switzerland",
    "belgium",
    "ireland",
    "portugal",
    "united states",
    "us",
}


@dataclass(frozen=True)
class Box:
    slide_number: int
    index: int
    text: str


@dataclass(frozen=True)
class PairCandidate:
    label: str
    value: str
    pattern: str
    box: Box
    table_id: str | None = None
    table_context_label: str | None = None
    table_axis: str | None = None


@dataclass
class MetricRecord:
    id: str
    name: str
    raw_value: str
    normalized_value: float | None
    unit: str | None
    metric_type: str
    category: str
    raw_context: str
    slide_number: int | None
    source: str
    table_id: str | None
    metric_catalog_id: int | None
    metric_catalog_slug: str | None
    extraction_confidence: float | None
    metadata: dict | None
    period_label: str | None
    period_start: str | None
    period_end: str | None
    brand: str | None
    baseline_text: str | None
    baseline_type: str | None


class BoxParser:
    def parse(self, path: Path) -> dict[int, list[Box]]:
        content = path.read_text(encoding="utf-8")
        parts = re.split(PAGE_SPLIT_RE, content)
        pages: dict[int, list[Box]] = {}
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                continue
            try:
                page_num = int(parts[i])
            except Exception:
                continue
            blocks = self._parse_box_blocks(parts[i + 1])
            pages[page_num] = [Box(slide_number=page_num, index=idx, text=txt) for idx, txt in enumerate(blocks)]
        return pages

    @staticmethod
    def _parse_box_blocks(page_content: str) -> list[str]:
        blocks: list[str] = []
        inside = False
        current: list[str] = []
        for raw in (page_content or "").splitlines():
            line = raw.rstrip()
            if line.strip() == DELIM_LINE:
                if not inside:
                    inside = True
                    current = []
                else:
                    text = "\n".join(current).strip()
                    if text:
                        blocks.append(text)
                    inside = False
                    current = []
                continue
            if inside:
                current.append(line)
        return blocks


class PairExtractor:
    def extract(self, box: Box) -> list[PairCandidate]:
        lines = [line.strip() for line in box.text.splitlines() if line.strip()]
        if not lines:
            return []

        pairs: list[PairCandidate] = []
        if lines[0].upper() == "[TABLE]":
            pairs.extend(self._extract_table_pairs(lines[1:], box))
        else:
            pairs.extend(self._extract_inline_pairs(lines, box))
            pairs.extend(self._extract_value_leading_inline_pairs(lines, box))
            pairs.extend(self._extract_stacked_pairs(lines, box))
            pairs.extend(self._extract_value_first_pairs(lines, box))

        out: list[PairCandidate] = []
        seen: set[tuple[str, str]] = set()
        for p in pairs:
            key = (p.label.lower().strip(), p.value.strip())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _extract_table_pairs(self, body: Sequence[str], box: Box) -> list[PairCandidate]:
        pairs: list[PairCandidate] = []
        raw_rows: list[list[str]] = []
        for line in body:
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 2 or not any(cells):
                continue
            raw_rows.append(cells)

        if not raw_rows:
            return pairs
        normalized_rows = normalize_table_rows(raw_rows)
        table_id = compute_table_uuid(slide_number=box.slide_number, rows=normalized_rows)

        matrix_pairs = self._extract_table_matrix_pairs(raw_rows, box, table_id=table_id)
        if matrix_pairs:
            pairs.extend(matrix_pairs)
            return pairs

        for cells in raw_rows:
            left, right = cells[0], cells[1]
            if is_value_line(left) and not is_value_line(right) and is_metric_label_candidate(right):
                pairs.append(
                    PairCandidate(
                        label=right,
                        value=left,
                        pattern="table_pipe",
                        box=box,
                        table_id=table_id,
                    )
                )
            elif is_value_line(right) and not is_value_line(left) and is_metric_label_candidate(left):
                pairs.append(
                    PairCandidate(
                        label=left,
                        value=right,
                        pattern="table_pipe",
                        box=box,
                        table_id=table_id,
                    )
                )

        for i in range(len(raw_rows) - 1):
            a = " | ".join(raw_rows[i]).strip()
            b = " | ".join(raw_rows[i + 1]).strip()
            if is_value_line(a) and not is_value_line(b) and is_metric_label_candidate(b):
                pairs.append(
                    PairCandidate(
                        label=b,
                        value=a,
                        pattern="table_stacked",
                        box=box,
                        table_id=table_id,
                    )
                )
        return pairs

    def _extract_table_matrix_pairs(
        self,
        rows: Sequence[Sequence[str]],
        box: Box,
        *,
        table_id: str,
    ) -> list[PairCandidate]:
        if len(rows) < 2:
            return []
        width = max(len(r) for r in rows)
        if width < 3:
            return []

        normalized: list[list[str]] = []
        for row in rows:
            padded = [c.strip() for c in row] + [""] * (width - len(row))
            normalized.append(padded)

        header = normalized[0]
        data = normalized[1:]
        if not any(data):
            return []

        header_metric_score = sum(1 for c in header[1:] if is_metric_label_candidate(c) and looks_metric_like(c))
        first_col_metric_score = sum(
            1
            for r in data
            if is_metric_label_candidate(r[0]) and looks_metric_like(r[0]) and not is_value_line(r[0])
        )
        orientation = "row_metric" if first_col_metric_score >= max(1, header_metric_score) else "column_metric"

        pairs: list[PairCandidate] = []
        if orientation == "column_metric":
            metric_headers = [h.strip() for h in header[1:]]
            for row in data:
                row_context = row[0].strip()
                for col_idx, metric_name in enumerate(metric_headers, start=1):
                    if not is_metric_label_candidate(metric_name):
                        continue
                    val = row[col_idx].strip() if col_idx < len(row) else ""
                    if not is_value_line(val):
                        continue
                    pairs.append(
                        PairCandidate(
                            label=metric_name,
                            value=val,
                            pattern="table_matrix_column_metric",
                            box=box,
                            table_id=table_id,
                            table_context_label=row_context or None,
                            table_axis="row",
                        )
                    )
            return pairs

        # row_metric: first column is metric name, other columns are context series.
        series_headers = [h.strip() for h in header[1:]]
        for row in data:
            metric_name = row[0].strip()
            if not is_metric_label_candidate(metric_name):
                continue
            for col_idx, series_name in enumerate(series_headers, start=1):
                val = row[col_idx].strip() if col_idx < len(row) else ""
                if not is_value_line(val):
                    continue
                pairs.append(
                    PairCandidate(
                        label=metric_name,
                        value=val,
                        pattern="table_matrix_row_metric",
                        box=box,
                        table_id=table_id,
                        table_context_label=series_name or None,
                        table_axis="column",
                    )
                )
        return pairs

    def _extract_inline_pairs(self, lines: Sequence[str], box: Box) -> list[PairCandidate]:
        pairs: list[PairCandidate] = []
        for line in lines:
            m = re.match(r"^(?P<label>[^:]{2,120}?)(?:\s*:\s*|\s+\-\s+)(?P<value>.+)$", line)
            if not m:
                continue
            label = m.group("label").strip(" -")
            if not is_metric_label_candidate(label):
                continue
            rhs = (m.group("value") or "").strip()
            if is_value_line(rhs):
                pairs.append(PairCandidate(label=label, value=rhs, pattern="inline", box=box))
            else:
                primary = extract_primary_value(rhs)
                if primary:
                    pairs.append(PairCandidate(label=label, value=primary, pattern="inline_partial", box=box))
        return pairs

    def _extract_stacked_pairs(self, lines: Sequence[str], box: Box) -> list[PairCandidate]:
        pairs: list[PairCandidate] = []
        for i in range(len(lines) - 1):
            label = lines[i].rstrip(":").strip()
            value = lines[i + 1].strip()
            if label and value and is_value_line(value) and not is_value_line(label) and is_metric_label_candidate(label):
                pairs.append(PairCandidate(label=label, value=value, pattern="stacked", box=box))
        return pairs

    def _extract_value_leading_inline_pairs(self, lines: Sequence[str], box: Box) -> list[PairCandidate]:
        pairs: list[PairCandidate] = []
        for line in lines:
            text = (line or "").strip()
            if not text:
                continue
            # Pure value lines should be handled by stacked/value-first/cross-box paths.
            if is_value_line(text):
                continue
            m = PRIMARY_VALUE_RE.match(text)
            if not m:
                continue
            value = (m.group(0) or "").strip()
            rest = text[m.end() :].strip(" -:|")
            if not rest:
                continue
            label = sanitize_metric_name(rest)
            if not is_metric_label_candidate(label):
                continue
            # Avoid long sentence-like tails becoming metric labels.
            if is_sentence_like_label(label):
                continue
            if len(label) > 64:
                continue
            pairs.append(
                PairCandidate(
                    label=label,
                    value=value,
                    pattern="inline_value_leading",
                    box=box,
                )
            )
        return pairs

    def _extract_value_first_pairs(self, lines: Sequence[str], box: Box) -> list[PairCandidate]:
        pairs: list[PairCandidate] = []
        for i in range(len(lines) - 1):
            first = lines[i].strip()
            second = lines[i + 1].strip()
            if first and second and is_value_line(first) and not is_value_line(second) and is_metric_label_candidate(second):
                pairs.append(PairCandidate(label=second, value=first, pattern="value_first", box=box))
        return pairs


class DescriptorResolver:
    def resolve(self, pair: PairCandidate, page_boxes: Sequence[Box]) -> tuple[str, dict | None]:
        label = pair.label.strip()
        if not is_country_label(label):
            return label, None

        descriptor = self._find_metric_descriptor(page_boxes, pair.box.index)
        if not descriptor:
            descriptor = self._find_page_metric_descriptor(page_boxes)

        if descriptor:
            return (
                descriptor,
                {
                    "country": label,
                    "original_label": label,
                    "name_resolved_from": "nearest_descriptor",
                },
            )
        return label, None

    def _find_metric_descriptor(self, boxes: Sequence[Box], current_index: int) -> str | None:
        ranked: list[tuple[float, str]] = []
        for radius in range(1, 7):
            for idx in (current_index - radius, current_index + radius):
                if idx < 0 or idx >= len(boxes):
                    continue
                candidate = boxes[idx].text
                if is_table_block(candidate) or is_source_block(candidate):
                    continue
                if not is_descriptor_block(candidate):
                    continue
                base = title_metric_base(candidate)
                if not base:
                    continue
                score = descriptor_candidate_score(base, distance=radius)
                ranked.append((score, base))
        if not ranked:
            return None
        best_score, best = max(ranked, key=lambda t: t[0])
        return best if best_score >= 0.15 else None

    def _find_page_metric_descriptor(self, boxes: Sequence[Box]) -> str | None:
        ranked: list[tuple[float, str]] = []
        for box in boxes:
            candidate = box.text
            if is_table_block(candidate) or is_source_block(candidate):
                continue
            if not is_descriptor_block(candidate):
                continue
            base = title_metric_base(candidate)
            if base:
                score = descriptor_candidate_score(base, distance=0)
                ranked.append((score, base))
        if not ranked:
            return None
        best_score, best = max(ranked, key=lambda t: t[0])
        return best if best_score >= 0.15 else None


class MetricScorer:
    _PATTERN_BASE = {
        "inline": 0.95,
        "inline_value_leading": 0.9,
        "inline_partial": 0.82,
        "stacked": 0.85,
        "table_pipe": 0.84,
        "table_stacked": 0.78,
        "table_matrix_row_metric": 0.9,
        "table_matrix_column_metric": 0.9,
        "value_first": 0.75,
        "cross_box": 0.62,
    }

    def score(
        self,
        pair: PairCandidate,
        name: str,
        unit: str | None,
        has_metadata: bool,
        *,
        metric_like: bool,
    ) -> tuple[float, dict]:
        score = self._PATTERN_BASE.get(pair.pattern, 0.6)
        components = {"pattern": score}

        if metric_like:
            score += 0.08
            components["metric_hint"] = 0.08

        if unit == "time" and any(token in name.lower() for token in ("duration", "time", "session")):
            score += 0.06
            components["unit_name_compat"] = 0.06
        elif unit == "percent" and any(token in name.lower() for token in ("rate", "lift", "share", "ctr", "cpa", "cpe")):
            score += 0.04
            components["unit_name_compat"] = 0.04

        if has_metadata:
            score += 0.04
            components["country_resolved"] = 0.04

        score = max(0.0, min(1.0, score))
        components["final"] = score
        return score, components


class MetricNameScorer:
    def score(
        self,
        *,
        metric_name: str,
        metric_like: bool,
        occurrence_count: int,
        similar_support_count: int,
    ) -> tuple[float, dict[str, float]]:
        name = (metric_name or "").strip()
        tokens = [t for t in re.findall(r"[a-z0-9]+", name.lower()) if t]
        score = 0.78
        components: dict[str, float] = {"base": score}

        if len(name) > 90:
            score -= 0.38
            components["length_penalty"] = -0.38
        elif len(name) > 70:
            score -= 0.28
            components["length_penalty"] = -0.28
        elif len(name) > 55:
            score -= 0.20
            components["length_penalty"] = -0.20
        elif len(name) > 45:
            score -= 0.12
            components["length_penalty"] = -0.12
        elif len(name) > 35:
            score -= 0.05
            components["length_penalty"] = -0.05

        if len(tokens) > 14:
            score -= 0.30
            components["token_penalty"] = -0.30
        elif len(tokens) > 10:
            score -= 0.20
            components["token_penalty"] = -0.20
        elif len(tokens) > 8:
            score -= 0.22
            components["token_penalty"] = -0.22
        elif len(tokens) > 6:
            score -= 0.08
            components["token_penalty"] = -0.08

        if is_sentence_like_label(name):
            score -= 0.22
            components["sentence_penalty"] = -0.22

        if metric_like:
            score += 0.06
            components["metric_hint_bonus"] = 0.06
        else:
            score -= 0.05
            components["metric_hint_penalty"] = -0.05

        if occurrence_count <= 1:
            score -= 0.10
            components["singleton_penalty"] = -0.10
        elif occurrence_count >= 3:
            score += 0.04
            components["occurrence_bonus"] = 0.04

        if similar_support_count <= 1:
            score -= 0.06
            components["low_support_penalty"] = -0.06
        elif similar_support_count >= 5:
            score += 0.08
            components["support_bonus"] = 0.08
        elif similar_support_count >= 3:
            score += 0.04
            components["support_bonus"] = 0.04

        if name.isupper() and len(name) <= 5:
            score += 0.05
            components["acronym_bonus"] = 0.05

        score = max(0.0, min(1.0, score))
        components["final"] = score
        return score, components


class RawContextResolver:
    def resolve(
        self,
        *,
        pair: PairCandidate,
        metric_name: str,
        page_boxes: Sequence[Box],
        descriptor_text: str | None,
    ) -> tuple[str, dict]:
        candidates: list[tuple[str, str, float]] = []
        thin_box = is_thin_context_block(pair.box.text)
        table_like = pair.pattern.startswith("table_")
        descriptor_matches_metric = bool(
            descriptor_text and normalize_text(descriptor_text) == normalize_text(metric_name)
        )

        box_text = pair.box.text.strip()
        if box_text:
            if table_like:
                score = 0.40 if thin_box else 0.46
            else:
                score = 0.55 if thin_box else 0.65
            if pair.value in box_text:
                score += 0.10
            if has_keyword_overlap(metric_name, box_text):
                score += 0.08
            candidates.append(("box", box_text, min(score, 0.95)))

        if descriptor_text:
            merged = f"{descriptor_text.strip()}\n{box_text}".strip()
            if descriptor_matches_metric:
                score = 0.66 if thin_box else 0.70
            else:
                if table_like:
                    score = 0.86
                else:
                    score = 0.80 if thin_box else (0.74 if pair.pattern != "cross_box" else 0.82)
            if has_keyword_overlap(metric_name, descriptor_text):
                score += 0.08
            if pair.value in merged:
                score += 0.04
            candidates.append(("descriptor_plus_box", merged, min(score, 0.98)))

        richer = self._best_richer_descriptor(
            page_boxes=page_boxes,
            current_index=pair.box.index,
            metric_name=metric_name,
            current_descriptor=descriptor_text,
        )
        if richer:
            merged = f"{richer}\n{box_text}".strip()
            if table_like:
                score = 0.90
            else:
                score = 0.86 if thin_box or pair.pattern == "cross_box" else 0.72
            if has_keyword_overlap(metric_name, richer):
                score += 0.04
            if pair.value in merged:
                score += 0.03
            candidates.append(("richer_descriptor_plus_box", merged, min(score, 0.97)))

        nearby = self._nearest_context_box_text(page_boxes, pair.box.index)
        if nearby:
            merged = f"{nearby.strip()}\n{box_text}".strip()
            score = 0.60
            if has_keyword_overlap(metric_name, nearby):
                score += 0.07
            candidates.append(("nearby_plus_box", merged, min(score, 0.90)))

        if not candidates:
            return box_text, {"context_source": "box", "context_confidence": 0.0}

        best_source, best_text, best_score = sorted(candidates, key=lambda c: c[2], reverse=True)[0]
        final_text = ensure_metric_name_in_context(metric_name, best_text)
        if pair.pattern.startswith("table_") and descriptor_text:
            final_text = ensure_series_context_in_context(descriptor_text.strip(), final_text)
        if pair.table_context_label:
            final_text = ensure_series_context_in_context(pair.table_context_label, final_text)
        support_text = self._best_supporting_nearby_text(
            page_boxes=page_boxes,
            current_index=pair.box.index,
            metric_name=metric_name,
            current_descriptor=descriptor_text,
        )
        if support_text and normalize_text(support_text) not in normalize_text(final_text):
            final_text = f"{final_text}\n{support_text}".strip()
        dimension_labels = self._nearby_dimension_labels(
            page_boxes=page_boxes,
            current_index=pair.box.index,
            max_labels=2,
        )
        appended_dims: list[str] = []
        for dim in dimension_labels:
            if normalize_text(dim) in normalize_text(final_text):
                continue
            final_text = f"{final_text}\n{dim}".strip()
            appended_dims.append(dim)
        details = {
            "context_source": best_source,
            "context_confidence": round(best_score, 3),
            "context_candidates": [
                {"source": source, "score": round(score, 3)}
                for source, _, score in sorted(candidates, key=lambda c: c[2], reverse=True)
            ],
        }
        if appended_dims:
            details["context_dimensions"] = appended_dims
        return final_text, details

    def _nearest_context_box_text(self, boxes: Sequence[Box], index: int) -> str | None:
        for radius in range(1, 4):
            for idx in (index - radius, index + radius):
                if idx < 0 or idx >= len(boxes):
                    continue
                candidate = boxes[idx].text
                if is_source_block(candidate) or is_table_block(candidate):
                    continue
                if not candidate.strip():
                    continue
                return candidate
        return None

    def _best_richer_descriptor(
        self,
        *,
        page_boxes: Sequence[Box],
        current_index: int,
        metric_name: str,
        current_descriptor: str | None,
    ) -> str | None:
        ranked: list[tuple[float, str]] = []
        metric_norm = normalize_text(metric_name)
        current_norm = normalize_text(current_descriptor or "")
        for box in page_boxes:
            text = box.text.strip()
            if not text or is_source_block(text) or is_table_block(text):
                continue
            if is_value_only_box(text):
                continue
            first = first_non_empty_line(text) or ""
            if not first:
                continue
            first_norm = normalize_text(first)
            if first_norm in {metric_norm, current_norm}:
                continue
            if looks_dimension_like_label(first):
                continue
            if len(first) < 18:
                continue
            distance = abs(box.index - current_index)
            score = descriptor_candidate_score(first, distance=distance)
            if score < 0.18:
                continue
            ranked.append((score, first))
        if not ranked:
            return None
        return max(ranked, key=lambda t: t[0])[1]

    def _best_supporting_nearby_text(
        self,
        *,
        page_boxes: Sequence[Box],
        current_index: int,
        metric_name: str,
        current_descriptor: str | None,
    ) -> str | None:
        ranked: list[tuple[float, str]] = []
        metric_norm = normalize_text(metric_name)
        descriptor_norm = normalize_text(current_descriptor or "")
        for radius in range(1, 5):
            for idx in (current_index - radius, current_index + radius):
                if idx < 0 or idx >= len(page_boxes):
                    continue
                text = page_boxes[idx].text.strip()
                if not text or is_source_block(text) or is_table_block(text) or is_value_only_box(text):
                    continue
                first = first_non_empty_line(text) or ""
                if not first:
                    continue
                first_norm = normalize_text(first)
                if first_norm in {metric_norm, descriptor_norm}:
                    continue
                if looks_dimension_like_label(first):
                    continue

                score = max(0.0, 0.22 - (0.04 * radius))
                if has_keyword_overlap(metric_name, text):
                    score += 0.10
                if is_sentence_like_label(first):
                    score += 0.12
                if len(first) >= 22:
                    score += 0.06
                if len(first) > 180:
                    score -= 0.12
                if score >= 0.18:
                    ranked.append((score, text))
        if not ranked:
            return None
        return max(ranked, key=lambda t: t[0])[1]

    def _nearby_dimension_labels(
        self,
        *,
        page_boxes: Sequence[Box],
        current_index: int,
        max_labels: int = 2,
    ) -> list[str]:
        labels: list[str] = []
        for radius in range(1, 6):
            for idx in (current_index - radius, current_index + radius):
                if idx < 0 or idx >= len(page_boxes):
                    continue
                text = page_boxes[idx].text.strip()
                if not text or is_source_block(text) or is_table_block(text) or is_value_only_box(text):
                    continue
                first = first_non_empty_line(text) or ""
                if not first:
                    continue
                if looks_dimension_like_label(first):
                    labels.append(first)
                    if len(labels) >= max_labels:
                        return labels
        return labels


class UnfilteredMetricsExtractor:
    def __init__(
        self,
        *,
        review_threshold: float = 0.65,
        database_url: str | None = None,
        catalog_hint_terms: set[str] | None = None,
    ) -> None:
        self._review_threshold = review_threshold
        self._parser = BoxParser()
        self._pair_extractor = PairExtractor()
        self._resolver = DescriptorResolver()
        self._scorer = MetricScorer()
        self._name_scorer = MetricNameScorer()
        self._context_resolver = RawContextResolver()
        if catalog_hint_terms is not None:
            self._catalog_hint_terms = {normalize_metric_name(term) for term in catalog_hint_terms if term}
        elif database_url:
            self._catalog_hint_terms = load_catalog_hint_terms_from_db(database_url)
        else:
            self._catalog_hint_terms = set()

    def extract(
        self,
        path: Path,
        *,
        require_metric_hint: bool = False,
        include_notes: bool = False,
    ) -> list[MetricRecord]:
        pages = self._parser.parse(path)
        pending: list[dict] = []

        for slide_number, boxes in sorted(pages.items()):
            page_descriptor = self._resolver._find_page_metric_descriptor(boxes)
            overall_slide_title = find_overall_slide_title(boxes)
            is_overall_slide = overall_slide_title is not None
            for box in boxes:
                if (not include_notes) and "### notes:" in box.text.lower():
                    continue
                if is_source_block(box.text):
                    continue

                pairs = self._pair_extractor.extract(box)
                if not pairs and is_value_only_box(box.text):
                    value = first_non_empty_line(box.text) or ""
                    descriptor = self._resolver._find_metric_descriptor(boxes, box.index) or page_descriptor
                    if descriptor and value:
                        pairs = [PairCandidate(label=descriptor, value=value, pattern="cross_box", box=box)]
                for pair in pairs:
                    name, metadata = self._resolver.resolve(pair, boxes)
                    name = sanitize_metric_name(name)
                    metric_like = self._is_metric_like(name)
                    if require_metric_hint and not metric_like:
                        continue
                    pending.append(
                        {
                            "slide_number": slide_number,
                            "boxes": boxes,
                            "pair": pair,
                            "name": name,
                            "metadata": metadata,
                            "metric_like": metric_like,
                            "is_overall_slide": is_overall_slide,
                            "overall_slide_title": overall_slide_title,
                        }
                    )

        name_counts, name_support = self._build_name_support(pending)
        records: list[MetricRecord] = []

        for item in pending:
            pair: PairCandidate = item["pair"]
            name: str = item["name"]
            metadata = item["metadata"]
            boxes: Sequence[Box] = item["boxes"]
            slide_number: int = item["slide_number"]
            metric_like: bool = bool(item.get("metric_like"))
            is_overall_slide: bool = bool(item.get("is_overall_slide"))
            overall_slide_title: str | None = item.get("overall_slide_title")

            normalized_value, unit = normalize_value(pair.value)
            pair_score, pair_components = self._scorer.score(
                pair,
                name=name,
                unit=unit,
                has_metadata=metadata is not None,
                metric_like=metric_like,
            )
            normalized_name = normalize_metric_name(name)
            occurrence_count = name_counts.get(normalized_name, 1)
            support_count = name_support.get(normalized_name, occurrence_count)
            name_score, name_components = self._name_scorer.score(
                metric_name=name,
                metric_like=metric_like,
                occurrence_count=occurrence_count,
                similar_support_count=support_count,
            )
            final_score, blend_components = self._blend_confidence(
                pair_score=pair_score,
                name_score=name_score,
                metric_name=name,
            )

            merged_meta = dict(metadata or {})
            merged_meta["pattern"] = pair.pattern
            merged_meta["is_overall_metric"] = is_overall_slide
            if is_overall_slide:
                merged_meta["overall_slide_title"] = overall_slide_title
                merged_meta["overall_slide_number"] = slide_number
            if pair.table_id:
                merged_meta["table_id"] = pair.table_id
            if pair.table_context_label:
                merged_meta["table_context_label"] = pair.table_context_label
            if pair.table_axis:
                merged_meta["table_axis"] = pair.table_axis
            merged_meta["name_occurrence_count"] = occurrence_count
            merged_meta["name_support_count"] = support_count
            merged_meta["confidence_components"] = {
                "pair": pair_components,
                "name": name_components,
                "blend": blend_components,
            }
            merged_meta["review_recommended"] = final_score < self._review_threshold

            descriptor_text = None
            if pair.pattern == "cross_box" or (metadata or {}).get("name_resolved_from"):
                descriptor_text = (
                    self._resolver._find_metric_descriptor(boxes, pair.box.index)
                    or self._resolver._find_page_metric_descriptor(boxes)
                )
            elif pair.pattern.startswith("table_"):
                descriptor_text = (
                    self._resolver._find_metric_descriptor(boxes, pair.box.index)
                    or self._resolver._find_page_metric_descriptor(boxes)
                )
            elif is_thin_context_block(pair.box.text):
                descriptor_text = (
                    self._resolver._find_metric_descriptor(boxes, pair.box.index)
                    or self._resolver._find_page_metric_descriptor(boxes)
                )
            chosen_context, context_meta = self._context_resolver.resolve(
                pair=pair,
                metric_name=name,
                page_boxes=boxes,
                descriptor_text=descriptor_text,
            )
            merged_meta.update(context_meta)

            records.append(
                MetricRecord(
                    id=str(uuid.uuid4()),
                    name=name,
                    raw_value=pair.value,
                    normalized_value=normalized_value,
                    unit=unit,
                    metric_type=metric_type_from_unit(unit),
                    category=category_from_label(name),
                    raw_context=chosen_context,
                    slide_number=slide_number,
                    source="box_raw",
                    table_id=pair.table_id,
                    metric_catalog_id=None,
                    metric_catalog_slug=None,
                    extraction_confidence=final_score,
                    metadata=merged_meta,
                    period_label=None,
                    period_start=None,
                    period_end=None,
                    brand=None,
                    baseline_text=None,
                    baseline_type=None,
                )
            )
        return records

    def _blend_confidence(
        self,
        *,
        pair_score: float,
        name_score: float,
        metric_name: str,
    ) -> tuple[float, dict[str, float]]:
        blend = (pair_score * 0.75) + (name_score * 0.25)
        components: dict[str, float] = {
            "pair_weight": 0.75,
            "name_weight": 0.25,
        }

        length = len((metric_name or "").strip())
        token_count = len(re.findall(r"[a-z0-9]+", (metric_name or "").lower()))
        penalty = 0.0
        if length > 90:
            penalty += 0.18
            components["very_long_name_penalty"] = -0.18
        elif length > 70:
            penalty += 0.12
            components["long_name_penalty"] = -0.12
        elif length > 55:
            penalty += 0.07
            components["long_name_penalty"] = -0.07

        if token_count > 14:
            penalty += 0.14
            components["very_long_token_penalty"] = -0.14
        elif token_count > 10:
            penalty += 0.08
            components["long_token_penalty"] = -0.08

        if is_sentence_like_label(metric_name):
            penalty += 0.08
            components["sentence_like_penalty"] = -0.08

        if penalty > 0:
            blend -= penalty
            components["penalty_total"] = -penalty

        final = max(0.0, min(1.0, blend))
        components["final"] = final
        return final, components

    def _is_metric_like(self, name: str) -> bool:
        if looks_metric_like(name):
            return True
        if not self._catalog_hint_terms:
            return False
        norm = normalize_metric_name(name)
        if not norm:
            return False
        padded = f" {norm} "
        if norm in self._catalog_hint_terms:
            return True
        for term in self._catalog_hint_terms:
            if len(term) < 3:
                continue
            if f" {term} " in padded:
                return True
        return False

    def _build_name_support(self, pending: Sequence[dict]) -> tuple[dict[str, int], dict[str, int]]:
        normalized_names = [normalize_metric_name(str(item["name"])) for item in pending if str(item["name"]).strip()]
        counts = Counter(normalized_names)
        support: dict[str, int] = {}
        names = list(counts.keys())
        for name in names:
            total = 0
            for other in names:
                if similar_metric_names(name, other):
                    total += counts[other]
            support[name] = total
        return dict(counts), support


def is_value_line(line: str) -> bool:
    return bool(VALUE_LINE_RE.fullmatch((line or "").strip()))


def extract_primary_value(text: str) -> str | None:
    match = PRIMARY_VALUE_RE.search((text or "").strip())
    return match.group(0).strip() if match else None


def normalize_value(raw_value: str) -> tuple[float | None, str | None]:
    text = (raw_value or "").strip()
    tmatch = TIME_RE.search(text)
    if tmatch:
        value = float(tmatch.group("num"))
        unit = tmatch.group("unit").lower()
        if unit in {"h", "hr", "hrs", "hour", "hours"}:
            value *= 60.0
        if unit in {"s", "sec", "secs", "second", "seconds"}:
            value /= 60.0
        return value, "time"

    rmatch = RATIO_RE.search(text)
    if rmatch:
        return float(rmatch.group("num")), "ratio"

    m = NUMBER_RE.search(text)
    if not m:
        return None, None
    value = float(m.group("num").replace(",", ""))
    suffix = (m.group("suffix") or "").strip().lower()
    if suffix == "k":
        value *= 1_000
    elif suffix == "m":
        value *= 1_000_000
    elif suffix == "b":
        value *= 1_000_000_000

    if "%" in text:
        return value, "percent"
    if "$" in text:
        return value, "currency"
    if "x" in text.lower():
        return value, "ratio"
    return value, "count"


def metric_type_from_unit(unit: str | None) -> str:
    if unit == "percent":
        return "percentage"
    if unit == "currency":
        return "currency"
    if unit == "time":
        return "time"
    if unit == "ratio":
        return "ratio"
    return "count"


def category_from_label(label: str) -> str:
    l = (label or "").lower()
    if any(k in l for k in ("cpa", "cpe", "cpi", "cpc", "cpm", "cpv", "cost", "spend", "investment")):
        return "cost"
    if any(k in l for k in ("reach", "impression", "sov", "frequency", "mau", "dau", "grp", "trp", "footprint")):
        return "reach"
    if any(k in l for k in ("session", "duration", "engagement", "engaged", "time spent")):
        return "engagement"
    if any(k in l for k in ("install", "launch", "click", "view", "conversion", "acquisition", "purchase", "rate", "lift", "growth")):
        return "performance"
    return "other"


def looks_metric_like(label: str) -> bool:
    return bool(METRIC_HINT_RE.search(label or ""))


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def is_country_label(label: str) -> bool:
    return normalize_text(label) in COUNTRY_LABELS


def is_table_block(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return bool(lines and lines[0].upper() == "[TABLE]")


def is_source_block(text: str) -> bool:
    first = (text.strip().splitlines() or [""])[0].strip().lower()
    return first.startswith("source:")


def title_metric_base(text: str) -> str:
    line = (text or "").splitlines()[0].strip()
    m = re.match(r"^(?P<base>.+?)\s+by\s+.+$", line, flags=re.IGNORECASE)
    return m.group("base").strip(" -") if m else line.strip(" -")


def is_descriptor_block(block: str) -> bool:
    lines = [ln.strip() for ln in (block or "").splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0]
    if len(lines) >= 2 and is_value_line(lines[1]):
        return False
    if is_value_line(first):
        return False
    if extract_primary_value(first) and re.search(r"[:=]|[$%]", first):
        return False
    data_like = sum(1 for ln in lines if re.search(r":\s*.*\d", ln))
    if data_like >= 1:
        return False
    return True


def first_non_empty_line(text: str) -> str | None:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return None


def is_value_only_box(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) != 1:
        return False
    return is_value_line(lines[0])


def is_thin_context_block(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return True
    if len(lines) <= 2:
        return True
    chars = len(" ".join(lines))
    return chars < 35


def has_keyword_overlap(metric_name: str, text: str) -> bool:
    metric_tokens = set(re.findall(r"[a-z0-9]+", (metric_name or "").lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if not metric_tokens or not text_tokens:
        return False
    stop = {"the", "and", "for", "in", "of", "to", "a", "an", "overall", "vs"}
    metric_tokens = {t for t in metric_tokens if t not in stop}
    return bool(metric_tokens & text_tokens)


def is_metric_label_candidate(label: str) -> bool:
    value = (label or "").strip()
    if not value:
        return False
    # Metric names must contain letters; reject pure numeric labels like "2025".
    if not re.search(r"[A-Za-z]", value):
        return False
    norm = normalize_metric_name(value)
    if FISCAL_LABEL_RE.fullmatch(norm):
        return False
    # Reject year-like stand-alone labels and tiny noise fragments.
    if re.fullmatch(r"(19|20)\d{2}", norm):
        return False
    if len(norm) < 2:
        return False
    return True


def find_overall_slide_title(boxes: Sequence[Box]) -> str | None:
    for box in boxes:
        text = box.text.strip()
        if not text or is_source_block(text) or is_table_block(text):
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            continue
        # Check first lines so titles split across two lines are still detected.
        for candidate in lines[:3]:
            if OVERALL_SLIDE_RE.search(candidate):
                return lines[0]
    return None


def normalize_metric_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()


def similar_metric_names(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if a_tokens and b_tokens:
        jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
        if jaccard >= 0.6:
            return True
    ratio = SequenceMatcher(a=a, b=b).ratio()
    return ratio >= 0.78 or a in b or b in a


def ensure_metric_name_in_context(metric_name: str, context_text: str) -> str:
    name = (metric_name or "").strip()
    text = (context_text or "").strip()
    if not name:
        return text
    if normalize_text(name) in normalize_text(text):
        return text
    return f"{name}\n{text}".strip()


def ensure_series_context_in_context(series_label: str, context_text: str) -> str:
    label = (series_label or "").strip()
    text = (context_text or "").strip()
    if not label:
        return text
    if normalize_text(label) in normalize_text(text):
        return text
    return f"{text}\n{label}".strip()


def sanitize_metric_name(name: str) -> str:
    text = " ".join((name or "").split()).strip()
    if not text:
        return ""
    # Remove common footnote markers at the end: "*", "†", "‡", "[1]", "(1)".
    text = re.sub(r"\s*[\*\u2020\u2021]+\s*$", "", text)
    text = re.sub(r"\s*(?:\[\d+\]|\(\d+\))\s*$", "", text)
    # Remove trailing separator punctuation.
    text = re.sub(r"[\s:;,.|/-]+$", "", text)
    return text.strip()


def load_catalog_hint_terms_from_db(database_url: str) -> set[str]:
    terms: set[str] = set()
    sync_url = database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    try:
        engine = create_engine(sync_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT mc.name AS metric_name, ma.alias AS alias_name
                    FROM metric_catalog mc
                    LEFT JOIN metric_aliases ma ON ma.metric_id = mc.id
                    """
                )
            )
            for metric_name, alias_name in rows:
                for value in (metric_name, alias_name):
                    if not value:
                        continue
                    norm = normalize_metric_name(str(value))
                    if norm:
                        terms.add(norm)
    except Exception:
        return set()
    return terms


def is_sentence_like_label(label: str) -> bool:
    text = (label or "").strip().lower()
    if not text:
        return False
    tokens = re.findall(r"[a-z0-9]+", text)
    if len(tokens) < 10:
        return False
    markers = {
        "led",
        "versus",
        "across",
        "continue",
        "continues",
        "delivered",
        "opportunity",
        "overall",
        "markets",
    }
    return len(markers.intersection(tokens)) >= 2


def looks_dimension_like_label(label: str) -> bool:
    text = normalize_metric_name(label)
    if not text:
        return False
    if looks_metric_like(text):
        return False
    dimension_tokens = {
        "audience",
        "exposed",
        "market",
        "markets",
        "country",
        "countries",
        "segment",
        "group",
    }
    tokens = set(text.split())
    return bool(tokens & dimension_tokens)


def descriptor_candidate_score(label: str, *, distance: int) -> float:
    text = (label or "").strip()
    if not text:
        return -1.0
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    token_count = len(tokens)
    score = 0.0

    score += max(0.0, 0.24 - (0.03 * max(distance, 0)))

    if looks_metric_like(text):
        score += 0.38
    else:
        score -= 0.04

    if re.match(r"^(avg|average|total|cost|cpa|cpe|ctr|vtr|reach|impressions?|duration|frequency|installs?|launches?|growth|lift)\b", text, flags=re.IGNORECASE):
        score += 0.16

    if token_count > 12:
        score -= 0.22
    elif token_count > 8:
        score -= 0.08
    elif 2 <= token_count <= 6:
        score += 0.12
    elif token_count == 1 and len(text) <= 6:
        score -= 0.35

    if len(text) > 85:
        score -= 0.32
    elif len(text) > 65:
        score -= 0.22
    elif len(text) > 45:
        score -= 0.08

    if is_sentence_like_label(text):
        score -= 0.30
    if looks_dimension_like_label(text):
        score -= 0.22
    if ":" in text and not looks_metric_like(text):
        score -= 0.08

    return score


def records_to_dict(records: Iterable[MetricRecord]) -> list[dict]:
    return [asdict(record) for record in records]
