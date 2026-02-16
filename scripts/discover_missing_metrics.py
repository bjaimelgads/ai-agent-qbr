#!/usr/bin/env python3
"""Discover candidate metric labels missing from the metric catalog.

Supports:
- ``metrics-json`` source: scans ``04_metrics_extracted.json`` files.
- ``box-raw`` source: scans ``02b_raw_content_by_box.txt`` files and extracts
  deterministic label/value pairs from per-box content.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sqlalchemy import create_engine, text

repo_root = Path(__file__).resolve().parents[1]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from qbr_intelligence.metrics.catalog import build_metric_catalog


DEFAULT_DB_URL = "sqlite+aiosqlite:///qbr_intelligence.db"
WINDOW_CHARS = 96
MAX_CONTEXT_SAMPLES = 3

DELIM_SPLIT_RE = re.compile(r"[\n|;]+")
SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9%/+& -]+")
BAD_TOKEN_RE = re.compile(
    r"\b(?:fy\d{2,4}|q[1-4]|h[12]|jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
    r"source|speaker|notes?|slide|image|rid\d+|markets?|models?)\b",
    re.IGNORECASE,
)
METRIC_HINT_RE = re.compile(
    r"\b(?:"
    r"ctr|vtr|vcr|cvr|roas|roi|cpa|cpi|cpe|cpc|cpm|cpv|"
    r"rate|lift|reach|impression|click|view|complete|spend|acquisition|conversion|purchase|"
    r"engagement|session|duration|frequency|install|launch|mau|dau|sov|grps?|trps?|"
    r"time spent|open rate|streaming hours|video starts?|viewability|fill rate|win rate"
    r")\b",
    re.IGNORECASE,
)
VALUE_LINE_RE = re.compile(
    r"^\s*(?:\$)?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*[KMBkmb])?(?:%|x|"
    r"\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))?\s*$",
    re.IGNORECASE,
)
VALUE_IN_LINE_RE = re.compile(
    r"(?:\$)?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*[KMBkmb])?(?:%|x|"
    r"\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))?",
    re.IGNORECASE,
)
PAGE_SPLIT_RE = re.compile(r"<!-- PAGE (\d+) -->")
DELIM_LINE = "- - - -"


@dataclass
class UnknownCandidate:
    label: str
    occurrences: int = 0
    files: set[str] = field(default_factory=set)
    slides: set[str] = field(default_factory=set)
    metric_types: Counter = field(default_factory=Counter)
    sample_values: list[str] = field(default_factory=list)
    sample_contexts: list[str] = field(default_factory=list)

    def add(
        self,
        *,
        file_name: str,
        slide_number: int | None,
        metric_type: str | None,
        value: str | None,
        context: str | None,
    ) -> None:
        self.occurrences += 1
        self.files.add(file_name)
        self.slides.add(f"{file_name}#slide-{slide_number}")
        if metric_type:
            self.metric_types[metric_type] += 1
        if value and value not in self.sample_values and len(self.sample_values) < MAX_CONTEXT_SAMPLES:
            self.sample_values.append(value)
        snippet = _compact_text(context)
        if snippet and snippet not in self.sample_contexts and len(self.sample_contexts) < MAX_CONTEXT_SAMPLES:
            self.sample_contexts.append(snippet)


@dataclass(frozen=True)
class CatalogMatcher:
    exact_labels: set[str]
    regex_patterns: tuple[re.Pattern[str], ...]

    def matches(self, candidate: str) -> bool:
        norm = normalize_label(candidate)
        if not norm:
            return False
        if norm in self.exact_labels:
            return True
        return any(p.search(candidate) for p in self.regex_patterns)


def _compact_text(text: str | None) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", text).strip()


def normalize_label(label: str) -> str:
    text = (label or "").lower().strip()
    text = NON_WORD_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip(" -:,.\t")
    text = re.sub(r"\b\d+(?:\.\d+)?\b\s*$", "", text).strip()
    text = text.replace(" avg ", " average ")
    text = text.replace(" ctr", " click through rate") if text == "ctr" else text
    return text


def _load_catalog_matcher(database_url: str) -> CatalogMatcher:
    exact: set[str] = set()
    regexes: list[re.Pattern[str]] = []

    # Fallback code catalog first.
    for entry in build_metric_catalog():
        exact.add(normalize_label(entry.name))
        for alias in entry.aliases:
            exact.add(normalize_label(alias.alias))
            if alias.pattern:
                try:
                    regexes.append(re.compile(alias.pattern, re.IGNORECASE))
                except re.error:
                    pass

    # DB catalog overrides/extends when available.
    sync_url = database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    try:
        engine = create_engine(sync_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT mc.name, ma.alias, ma.pattern
                    FROM metric_catalog mc
                    LEFT JOIN metric_aliases ma ON ma.metric_id = mc.id
                    """
                )
            )
            for name, alias, pattern in rows:
                if name:
                    exact.add(normalize_label(name))
                if alias:
                    exact.add(normalize_label(alias))
                if pattern:
                    try:
                        regexes.append(re.compile(pattern, re.IGNORECASE))
                    except re.error:
                        continue
    except Exception:
        # Fallback-only mode is acceptable.
        pass

    exact.discard("")
    return CatalogMatcher(exact_labels=exact, regex_patterns=tuple(regexes))


def _split_tail_segment(text: str) -> str:
    parts = [p.strip() for p in DELIM_SPLIT_RE.split(text) if p.strip()]
    return parts[-1] if parts else text.strip()


def _split_head_segment(text: str) -> str:
    parts = [p.strip() for p in DELIM_SPLIT_RE.split(text) if p.strip()]
    return parts[0] if parts else text.strip()


def extract_candidate_labels(context: str, value: str) -> list[str]:
    context_text = context or ""
    value_text = (value or "").strip()
    if not context_text or not value_text:
        return []

    candidates: list[str] = []

    # Explicit "label : value" and "value : label" captures.
    value_re = re.escape(value_text)
    before_re = re.compile(rf"([A-Za-z][A-Za-z0-9/&+%\- ]{{2,80}}?)\s*[:=\-]\s*{value_re}", re.IGNORECASE)
    after_re = re.compile(rf"{value_re}\s*[:=\-]\s*([A-Za-z][A-Za-z0-9/&+%\- ]{{2,80}})", re.IGNORECASE)
    for m in before_re.finditer(context_text):
        candidates.append(m.group(1).strip())
    for m in after_re.finditer(context_text):
        candidates.append(m.group(1).strip())

    # Fallback around first occurrence window.
    idx = context_text.lower().find(value_text.lower())
    if idx >= 0:
        left = context_text[max(0, idx - WINDOW_CHARS) : idx]
        right = context_text[idx + len(value_text) : idx + len(value_text) + WINDOW_CHARS]

        left_seg = _split_tail_segment(left)
        right_seg = _split_head_segment(right)

        left_match = re.search(r"([A-Za-z][A-Za-z0-9/&+%\- ]{2,80})\s*$", left_seg)
        if left_match:
            candidates.append(left_match.group(1).strip())

        right_match = re.search(r"^\s*([A-Za-z][A-Za-z0-9/&+%\- ]{2,80})", right_seg)
        if right_match:
            candidates.append(right_match.group(1).strip())

    # De-duplicate preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        norm = normalize_label(candidate)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(candidate)
    return deduped


def is_noise_label(label: str) -> bool:
    norm = normalize_label(label)
    if not norm:
        return True
    if len(norm) < 4:
        return True
    if re.fullmatch(r"[0-9.\-+% ]+", norm):
        return True
    if BAD_TOKEN_RE.search(norm) and len(norm.split()) <= 2:
        return True
    if sum(ch.isalpha() for ch in norm) < 3:
        return True
    return False


def looks_like_metric_label(label: str) -> bool:
    return bool(METRIC_HINT_RE.search(label or ""))


def iter_metric_files(root: Path) -> Iterable[Path]:
    yield from root.glob("**/04_metrics_extracted.json")


def iter_box_files(root: Path) -> Iterable[Path]:
    yield from root.glob("**/02b_raw_content_by_box.txt")


def infer_metric_type(value: str) -> str:
    text = (value or "").lower()
    if "$" in text:
        return "currency"
    if "%" in text:
        return "percentage"
    if any(token in text for token in ("min", "hour", "hr", "sec")):
        return "time"
    if "x" in text:
        return "ratio"
    return "count"


def _is_value_line(line: str) -> bool:
    return bool(VALUE_LINE_RE.fullmatch((line or "").strip()))


def _extract_value_in_line(line: str) -> str | None:
    match = VALUE_IN_LINE_RE.search(line or "")
    if not match:
        return None
    return match.group(0).strip()


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


def _iter_page_blocks(path: Path) -> Iterable[tuple[int, str]]:
    content = path.read_text(encoding="utf-8")
    parts = re.split(PAGE_SPLIT_RE, content)
    for i in range(1, len(parts), 2):
        if i + 1 >= len(parts):
            continue
        try:
            page_num = int(parts[i])
        except Exception:
            continue
        page_content = parts[i + 1]
        for block in _parse_box_blocks(page_content):
            yield page_num, block


def _extract_pairs_from_box(block: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    lines = [line.strip() for line in (block or "").splitlines() if line.strip()]
    if not lines:
        return pairs

    # Table style block.
    if lines[0].upper() == "[TABLE]":
        body = lines[1:]
        for line in body:
            if "|" not in line:
                continue
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < 2:
                continue
            left, right = cells[0], cells[1]
            if _is_value_line(left) and not _is_value_line(right):
                pairs.append((right, left))
            elif _is_value_line(right) and not _is_value_line(left):
                pairs.append((left, right))
        return pairs

    # In-line pairs: Label: Value or Label - Value
    for line in lines:
        match = re.match(r"^(?P<label>[^:]{2,120}?)[\s]*[:\-][\s]*(?P<value>.+)$", line)
        if not match:
            continue
        label = match.group("label").strip(" -")
        value = (match.group("value") or "").strip()
        if _is_value_line(value):
            pairs.append((label, value))

    # Stacked pairs: label line followed by value line.
    for idx in range(len(lines) - 1):
        label = lines[idx].rstrip(":").strip()
        value = lines[idx + 1].strip()
        if not label or not value:
            continue
        if _is_value_line(value) and not _is_value_line(label):
            pairs.append((label, value))

    # De-dup pairs while preserving order.
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for label, value in pairs:
        key = (normalize_label(label), value.strip())
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append((label, value.strip()))
    return deduped


def discover_unknown_metrics(
    root: Path,
    matcher: CatalogMatcher,
    *,
    source: str,
    include_notes: bool,
    require_metric_hint: bool,
    ignore_catalog: bool,
    verbose: bool,
) -> dict[str, UnknownCandidate]:
    unknown: dict[str, UnknownCandidate] = {}

    paths = (
        sorted(iter_metric_files(root))
        if source == "metrics-json"
        else sorted(iter_box_files(root))
    )

    for path in paths:
        if verbose:
            print(f"[scan] {path}")

        file_name = str(path.relative_to(root))
        before_count = len(unknown)
        processed_items = 0

        if source == "metrics-json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                if verbose:
                    print("  - skipped (invalid JSON)")
                continue
            if not isinstance(payload, list):
                if verbose:
                    print("  - skipped (payload is not a list)")
                continue

            for item in payload:
                if not isinstance(item, dict):
                    continue
                processed_items += 1
                value = str(item.get("value") or "").strip()
                context = str(item.get("context") or "")
                slide_number = item.get("slide_number")
                metric_type = item.get("metric_type")
                if not include_notes and "### notes:" in context.lower():
                    continue

                candidates = extract_candidate_labels(context=context, value=value)
                for candidate in candidates:
                    if is_noise_label(candidate):
                        continue
                    if require_metric_hint and not looks_like_metric_label(candidate):
                        continue
                    if (not ignore_catalog) and matcher.matches(candidate):
                        continue

                    key = normalize_label(candidate)
                    bucket = unknown.get(key)
                    if bucket is None:
                        bucket = UnknownCandidate(label=key)
                        unknown[key] = bucket
                    bucket.add(
                        file_name=file_name,
                        slide_number=slide_number,
                        metric_type=metric_type,
                        value=value,
                        context=context,
                    )
        else:
            try:
                page_blocks = list(_iter_page_blocks(path))
            except Exception:
                if verbose:
                    print("  - skipped (cannot parse box file)")
                continue
            for slide_number, block in page_blocks:
                if not include_notes and "### notes:" in block.lower():
                    continue
                pairs = _extract_pairs_from_box(block)
                processed_items += len(pairs)
                for label, value in pairs:
                    if is_noise_label(label):
                        continue
                    if require_metric_hint and not looks_like_metric_label(label):
                        continue
                    if (not ignore_catalog) and matcher.matches(label):
                        continue

                    key = normalize_label(label)
                    bucket = unknown.get(key)
                    if bucket is None:
                        bucket = UnknownCandidate(label=key)
                        unknown[key] = bucket
                    bucket.add(
                        file_name=file_name,
                        slide_number=slide_number,
                        metric_type=infer_metric_type(value),
                        value=value,
                        context=block,
                    )
        if verbose:
            added = len(unknown) - before_count
            print(f"  - pairs scanned: {processed_items}, new unknown labels from file: {added}")

    return unknown


def write_csv(path: Path, unknown: dict[str, UnknownCandidate], *, min_occurrences: int) -> int:
    rows = [
        candidate
        for candidate in unknown.values()
        if candidate.occurrences >= min_occurrences
    ]
    rows.sort(key=lambda c: (c.occurrences, len(c.files), c.label), reverse=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_label",
                "occurrences",
                "deck_count",
                "metric_types",
                "sample_values",
                "sample_context_1",
                "sample_context_2",
                "sample_context_3",
                "sample_slides",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_label": row.label,
                    "occurrences": row.occurrences,
                    "deck_count": len(row.files),
                    "metric_types": "; ".join(
                        f"{k}:{v}" for k, v in row.metric_types.most_common()
                    ),
                    "sample_values": " | ".join(row.sample_values),
                    "sample_context_1": row.sample_contexts[0] if len(row.sample_contexts) > 0 else "",
                    "sample_context_2": row.sample_contexts[1] if len(row.sample_contexts) > 1 else "",
                    "sample_context_3": row.sample_contexts[2] if len(row.sample_contexts) > 2 else "",
                    "sample_slides": " | ".join(sorted(row.slides)[:MAX_CONTEXT_SAMPLES]),
                }
            )
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover missing metric labels from extraction outputs")
    parser.add_argument(
        "--root",
        default="extraction_output",
        help="Root directory that contains extraction_output deck folders",
    )
    parser.add_argument(
        "--source",
        choices=("metrics-json", "box-raw"),
        default="box-raw",
        help="Input source: 04_metrics_extracted.json or 02b_raw_content_by_box.txt",
    )
    parser.add_argument(
        "--out",
        default="artifacts/missing_metric_candidates.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DB_URL,
        help="Database URL used to read metric_catalog + metric_aliases",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Only keep candidates with at least this many occurrences",
    )
    parser.add_argument(
        "--include-notes",
        action="store_true",
        help="Include contexts that come from notes sections (default excludes notes)",
    )
    parser.add_argument(
        "--allow-non-metric-labels",
        action="store_true",
        help="Do not require metric-like keywords in candidate labels",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each input file and per-file summary",
    )
    parser.add_argument(
        "--ignore-catalog",
        action="store_true",
        help="Do not filter labels by metric_catalog/metric_aliases; keep all extracted metric labels",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    out = Path(args.out)

    matcher = _load_catalog_matcher(args.database_url)
    unknown = discover_unknown_metrics(
        root=root,
        matcher=matcher,
        source=args.source,
        include_notes=args.include_notes,
        require_metric_hint=not args.allow_non_metric_labels,
        ignore_catalog=args.ignore_catalog,
        verbose=args.verbose,
    )
    count = write_csv(out, unknown, min_occurrences=args.min_occurrences)

    print(f"Scanned root: {root}")
    print(f"Known catalog labels: {len(matcher.exact_labels)}")
    if args.ignore_catalog:
        print(f"Extracted metric label candidates: {len(unknown)}")
    else:
        print(f"Unknown candidates found: {len(unknown)}")
    print(f"CSV rows written (min-occurrences={args.min_occurrences}): {count}")
    print(f"Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
