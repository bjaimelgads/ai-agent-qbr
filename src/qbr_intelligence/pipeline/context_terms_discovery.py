"""Discover important non-metric context terms around metric mentions in deck outputs."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from sqlalchemy import create_engine, text

from qbr_intelligence.metrics.catalog import build_metric_catalog

PAGE_SPLIT_RE = re.compile(r"<!-- PAGE (\d+) -->")
DELIM_LINE = "- - - -"
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+&/-]*")
NON_WORD_RE = re.compile(r"[^a-z0-9+&/% -]+")

DEFAULT_METRIC_HINT_RE = re.compile(
    r"\b(?:"
    r"ctr|vtr|vcr|cvr|roas|roi|cpa|cpi|cpe|cpc|cpm|cpv|"
    r"rate|lift|reach|impression|click|view|complete|spend|acquisition|conversion|purchase|"
    r"engagement|session|duration|frequency|install|launch|mau|dau|sov|grps?|trps?|"
    r"time spent|open rate|streaming hours|video starts?|viewability|fill rate|win rate|"
    r"investment|acquired users?"
    r")\b",
    re.IGNORECASE,
)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "them", "to", "up", "was", "were",
    "with", "vs", "versus", "across", "over", "under", "than", "this", "these", "those", "any", "all",
    "can", "could", "would", "should", "may", "might", "will", "new", "fy", "h1", "h2", "q1", "q2", "q3", "q4",
    "source", "note", "notes", "question", "details", "reporting", "report", "metrics", "standard",
}

DOMAIN_TOKENS = {
    "installed", "not", "lapsed", "active", "audience", "segment", "roadblock", "roadblocks",
    "ros", "placement", "placements", "product", "products", "creative", "creatives",
    "carousel", "companion", "video", "static", "banner", "inline", "home", "screen",
    "sponsorship", "store", "content", "native", "unit", "units", "multicard", "multi-card",
    "countdown", "clock", "bundle", "standalone", "tentpole", "title", "av", "rb", "3d",
}

GENERIC_NOISE_TOKENS = {
    "data", "based", "more", "user", "users", "campaigns", "disney", "disney+", "lg", "solutions",
    "source", "overall", "total", "higher", "within", "media", "cost", "minute", "minutes", "paid",
    "market", "markets", "global", "domestic", "quarterly", "weekly", "analysis", "duration",
}

NOISE_TERMS = {
    "business review",
    "looking ahead",
    "partnership",
    "highlights",
    "section",
    "domestic",
    "global",
    "weekly",
    "quarterly",
}

TYPE_KEYWORDS: list[tuple[str, set[str]]] = [
    ("audience_segment", {"installed", "lapsed", "active", "intenders", "acquired", "new users", "not installed"}),
    ("placement", {"roadblock", "home screen", "content store", "ros", "sponsorship", "inline", "banner"}),
    ("ad_product", {"product", "package", "sponsorship", "roadblock", "rb", "companion"}),
    ("creative_unit", {"carousel", "video", "static", "creative", "cards", "countdown", "animation", "3d", "unit"}),
    ("reporting_or_measurement", {"kochava", "lookback", "attribution", "mmp", "dashboard", "reporting"}),
]


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


@dataclass(frozen=True)
class MetricHit:
    metric_id: str
    label: str


@dataclass
class ContextMention:
    term: str
    metric_id: str
    metric_label: str
    deck: str
    slide_number: int
    source_kind: str
    snippet: str


@dataclass
class ContextAggregate:
    term: str
    occurrences: int = 0
    decks: set[str] = field(default_factory=set)
    metrics: set[str] = field(default_factory=set)
    source_kinds: Counter[str] = field(default_factory=Counter)
    sample_snippets: list[str] = field(default_factory=list)

    def add(self, mention: ContextMention) -> None:
        self.occurrences += 1
        self.decks.add(mention.deck)
        self.metrics.add(mention.metric_id)
        self.source_kinds[mention.source_kind] += 1
        compact = compact_text(mention.snippet)
        if compact and compact not in self.sample_snippets and len(self.sample_snippets) < 3:
            self.sample_snippets.append(compact)


@dataclass
class MetricContextAggregate:
    metric_id: str
    metric_label: str
    term: str
    occurrences: int = 0
    decks: set[str] = field(default_factory=set)
    sample_snippets: list[str] = field(default_factory=list)

    def add(self, mention: ContextMention) -> None:
        self.occurrences += 1
        self.decks.add(mention.deck)
        compact = compact_text(mention.snippet)
        if compact and compact not in self.sample_snippets and len(self.sample_snippets) < 3:
            self.sample_snippets.append(compact)


def compact_text(text: str | None) -> str:
    return SPACE_RE.sub(" ", (text or "")).strip()


def normalize_label(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = NON_WORD_RE.sub(" ", lowered)
    lowered = SPACE_RE.sub(" ", lowered).strip(" -:,.;")
    return lowered


def _load_catalog_matcher(database_url: str) -> CatalogMatcher:
    exact: set[str] = set()
    regexes: list[re.Pattern[str]] = []

    for entry in build_metric_catalog():
        exact.add(normalize_label(entry.name))
        for alias in entry.aliases:
            if alias.alias:
                exact.add(normalize_label(alias.alias))
            if alias.pattern:
                try:
                    regexes.append(re.compile(alias.pattern, re.IGNORECASE))
                except re.error:
                    continue

    sync_url = database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    try:
        engine = create_engine(sync_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT mc.slug, mc.name, ma.alias, ma.pattern
                    FROM metric_catalog mc
                    LEFT JOIN metric_aliases ma ON ma.metric_id = mc.id
                    """
                )
            )
            for row in rows:
                name = row.name
                alias = row.alias
                pattern = row.pattern
                if name:
                    exact.add(normalize_label(str(name)))
                if alias:
                    exact.add(normalize_label(str(alias)))
                if pattern:
                    try:
                        regexes.append(re.compile(str(pattern), re.IGNORECASE))
                    except re.error:
                        continue
    except Exception:
        pass

    exact.discard("")
    return CatalogMatcher(exact_labels=exact, regex_patterns=tuple(regexes))


def _build_metric_phrase_index() -> dict[str, str]:
    phrase_to_metric: dict[str, str] = {}
    for entry in build_metric_catalog():
        phrase_to_metric[normalize_label(entry.name)] = entry.metric_id
        for alias in entry.aliases:
            if alias.alias:
                phrase_to_metric[normalize_label(alias.alias)] = entry.metric_id
    return {k: v for k, v in phrase_to_metric.items() if k}


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


def iter_box_files(root: Path) -> Iterable[Path]:
    yield from root.glob("**/02b_raw_content_by_box.txt")


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
        for block in _parse_box_blocks(parts[i + 1]):
            yield page_num, block


def _metric_hits_in_text(
    text: str,
    *,
    metric_phrases: dict[str, str],
    metric_matcher: CatalogMatcher,
) -> list[MetricHit]:
    lowered = normalize_label(text)
    hits: list[MetricHit] = []
    seen: set[str] = set()

    for phrase, metric_id in metric_phrases.items():
        if len(phrase) < 3:
            continue
        if phrase in lowered:
            if metric_id not in seen:
                hits.append(MetricHit(metric_id=metric_id, label=phrase))
                seen.add(metric_id)

    # fallback for cases where catalog regex catches it but phrase lookup misses
    if metric_matcher.matches(text) and not hits:
        hits.append(MetricHit(metric_id="unknown_metric", label="metric_match"))

    return hits


def _extract_table_cells(block: str) -> list[list[str]]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines or lines[0].upper() != "[TABLE]":
        return []

    rows: list[list[str]] = []
    for line in lines[1:]:
        if "|" not in line:
            continue
        row = [cell.strip() for cell in line.split("|")]
        if len(row) >= 2:
            rows.append(row)
    return rows


def _looks_numeric(text: str) -> bool:
    value = (text or "").strip()
    return bool(
        re.fullmatch(
            r"(?:\$)?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*[KMBkmb])?(?:%|x|\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))?",
            value,
            flags=re.IGNORECASE,
        )
    )


def _table_dimension_mentions(
    *,
    rows: list[list[str]],
    deck: str,
    slide_number: int,
    metric_phrases: dict[str, str],
    matcher: CatalogMatcher,
) -> list[ContextMention]:
    if not rows:
        return []

    width = max(len(r) for r in rows)
    if width < 3 or len(rows) < 2:
        return []

    normalized: list[list[str]] = []
    for row in rows:
        normalized.append(row + [""] * (width - len(row)))

    header = normalized[0]
    body = normalized[1:]
    mentions: list[ContextMention] = []

    # orientation: if first column looks metric-heavy -> row_metric; else column_metric
    first_col_metric = 0
    header_metric = 0
    for row in body:
        if _metric_hits_in_text(row[0], metric_phrases=metric_phrases, metric_matcher=matcher):
            first_col_metric += 1
    for cell in header[1:]:
        if _metric_hits_in_text(cell, metric_phrases=metric_phrases, metric_matcher=matcher):
            header_metric += 1
    row_metric = first_col_metric >= max(1, header_metric)

    if row_metric:
        series = header[1:]
        for row in body:
            metric_hits = _metric_hits_in_text(row[0], metric_phrases=metric_phrases, metric_matcher=matcher)
            if not metric_hits:
                continue
            for idx, series_label in enumerate(series, start=1):
                if idx >= len(row):
                    continue
                value = row[idx]
                if not _looks_numeric(value):
                    continue
                term = normalize_label(series_label)
                if not term or is_metric_or_noise_term(term, matcher=matcher):
                    continue
                for hit in metric_hits:
                    mentions.append(
                        ContextMention(
                            term=term,
                            metric_id=hit.metric_id,
                            metric_label=hit.label,
                            deck=deck,
                            slide_number=slide_number,
                            source_kind="table_dimension",
                            snippet=f"{row[0]} | {series_label} | {value}",
                        )
                    )
        return mentions

    metric_headers = header[1:]
    for row in body:
        series_label = normalize_label(row[0])
        if not series_label:
            continue
        if is_metric_or_noise_term(series_label, matcher=matcher):
            continue
        for idx, metric_cell in enumerate(metric_headers, start=1):
            if idx >= len(row):
                continue
            value = row[idx]
            if not _looks_numeric(value):
                continue
            metric_hits = _metric_hits_in_text(metric_cell, metric_phrases=metric_phrases, metric_matcher=matcher)
            if not metric_hits:
                continue
            for hit in metric_hits:
                mentions.append(
                    ContextMention(
                        term=series_label,
                        metric_id=hit.metric_id,
                        metric_label=hit.label,
                        deck=deck,
                        slide_number=slide_number,
                        source_kind="table_dimension",
                        snippet=f"{metric_cell} | {row[0]} | {value}",
                    )
                )
    return mentions


def is_metric_or_noise_term(term: str, *, matcher: CatalogMatcher) -> bool:
    norm = normalize_label(term)
    if not norm:
        return True
    if norm in NOISE_TERMS:
        return True
    if matcher.matches(norm):
        return True
    if DEFAULT_METRIC_HINT_RE.search(norm):
        return True
    if any(ch.isdigit() for ch in norm):
        return True
    tokens = [t for t in norm.split() if t]
    if not tokens:
        return True
    if all(t in STOPWORDS for t in tokens):
        return True
    if len(tokens) == 1 and (len(tokens[0]) < 3 or tokens[0] in STOPWORDS):
        return True
    if len(tokens) == 1 and tokens[0] in GENERIC_NOISE_TOKENS and tokens[0] not in DOMAIN_TOKENS:
        return True
    return False


def _candidate_terms_from_text(
    text: str,
    *,
    matcher: CatalogMatcher,
    max_ngram: int = 4,
) -> list[str]:
    raw_tokens = [tok.lower() for tok in TOKEN_RE.findall(text or "")]
    tokens: list[str] = []
    for tok in raw_tokens:
        norm_tok = normalize_label(tok)
        if not norm_tok or norm_tok in STOPWORDS:
            continue
        if re.fullmatch(r"[a-z]?\d+[a-z]?", norm_tok):
            continue
        tokens.append(norm_tok)

    candidates: list[str] = []
    for size in range(1, max_ngram + 1):
        for i in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[i : i + size])
            if is_metric_or_noise_term(phrase, matcher=matcher):
                continue
            phrase_tokens = phrase.split()
            has_domain_token = any(tok in DOMAIN_TOKENS for tok in phrase_tokens)
            if size == 1:
                if not has_domain_token:
                    continue
            else:
                if not has_domain_token and not any(" " in kw and kw in phrase for kw in DOMAIN_TOKENS):
                    continue
            if size == 1 and len(phrase) < 4:
                continue
            if len(phrase_tokens) > 4:
                continue
            candidates.append(phrase)

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in candidates:
        if phrase in seen:
            continue
        seen.add(phrase)
        deduped.append(phrase)
    return deduped


def _narrative_mentions(
    *,
    block: str,
    deck: str,
    slide_number: int,
    metric_phrases: dict[str, str],
    matcher: CatalogMatcher,
) -> list[ContextMention]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return []

    mentions: list[ContextMention] = []
    for idx, line in enumerate(lines):
        metric_hits = _metric_hits_in_text(line, metric_phrases=metric_phrases, metric_matcher=matcher)
        if not metric_hits:
            continue
        lo = max(0, idx - 1)
        hi = min(len(lines), idx + 2)
        window = " ".join(lines[lo:hi])
        terms = _candidate_terms_from_text(window, matcher=matcher)
        for term in terms:
            for hit in metric_hits:
                mentions.append(
                    ContextMention(
                        term=term,
                        metric_id=hit.metric_id,
                        metric_label=hit.label,
                        deck=deck,
                        slide_number=slide_number,
                        source_kind="narrative_window",
                        snippet=window,
                    )
                )
    return mentions


def infer_topic(term: str) -> str:
    value = normalize_label(term)
    for topic, keywords in TYPE_KEYWORDS:
        for keyword in keywords:
            if keyword in value:
                return topic
    return "other"


def score_context(aggregate: ContextAggregate) -> float:
    table_mentions = aggregate.source_kinds.get("table_dimension", 0)
    narrative_mentions = aggregate.source_kinds.get("narrative_window", 0)
    score = (
        (table_mentions * 2.5)
        + (narrative_mentions * 1.0)
        + (len(aggregate.decks) * 1.5)
        + (len(aggregate.metrics) * 1.2)
        + (aggregate.occurrences * 0.2)
    )
    return round(score, 4)


def _is_high_signal_term(term: str) -> bool:
    tokens = term.split()
    if not tokens:
        return False
    has_domain = any(tok in DOMAIN_TOKENS for tok in tokens)
    if has_domain:
        return True
    # Keep compact multi-token proper context-like phrases.
    return len(tokens) >= 2 and len(tokens) <= 3


def discover_metric_context_terms(
    *,
    root: Path,
    database_url: str,
    min_occurrences: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matcher = _load_catalog_matcher(database_url)
    metric_phrases = _build_metric_phrase_index()

    aggregate: dict[str, ContextAggregate] = {}
    by_metric: dict[tuple[str, str], MetricContextAggregate] = {}

    for path in sorted(iter_box_files(root)):
        deck = str(path.parent.name)
        for slide_number, block in _iter_page_blocks(path):
            table_rows = _extract_table_cells(block)
            mentions = _table_dimension_mentions(
                rows=table_rows,
                deck=deck,
                slide_number=slide_number,
                metric_phrases=metric_phrases,
                matcher=matcher,
            )
            mentions.extend(
                _narrative_mentions(
                    block=block,
                    deck=deck,
                    slide_number=slide_number,
                    metric_phrases=metric_phrases,
                    matcher=matcher,
                )
            )

            for mention in mentions:
                if is_metric_or_noise_term(mention.term, matcher=matcher):
                    continue
                bucket = aggregate.get(mention.term)
                if bucket is None:
                    bucket = ContextAggregate(term=mention.term)
                    aggregate[mention.term] = bucket
                bucket.add(mention)

                metric_key = (mention.metric_id, mention.term)
                metric_bucket = by_metric.get(metric_key)
                if metric_bucket is None:
                    metric_bucket = MetricContextAggregate(
                        metric_id=mention.metric_id,
                        metric_label=mention.metric_label,
                        term=mention.term,
                    )
                    by_metric[metric_key] = metric_bucket
                metric_bucket.add(mention)

    ranked_contexts: list[dict[str, object]] = []
    for row in aggregate.values():
        if row.occurrences < min_occurrences:
            continue
        if not _is_high_signal_term(row.term):
            continue
        ranked_contexts.append(
            {
                "context_term": row.term,
                "topic": infer_topic(row.term),
                "importance_score": score_context(row),
                "occurrences": row.occurrences,
                "deck_count": len(row.decks),
                "metric_count": len(row.metrics),
                "source_table_dimension": row.source_kinds.get("table_dimension", 0),
                "source_narrative_window": row.source_kinds.get("narrative_window", 0),
                "sample_context_1": row.sample_snippets[0] if len(row.sample_snippets) > 0 else "",
                "sample_context_2": row.sample_snippets[1] if len(row.sample_snippets) > 1 else "",
                "sample_context_3": row.sample_snippets[2] if len(row.sample_snippets) > 2 else "",
            }
        )

    ranked_contexts.sort(
        key=lambda r: (
            float(r["importance_score"]),
            int(r["deck_count"]),
            int(r["metric_count"]),
            int(r["occurrences"]),
            str(r["context_term"]),
        ),
        reverse=True,
    )

    metric_rows: list[dict[str, object]] = []
    for row in by_metric.values():
        if row.occurrences < min_occurrences:
            continue
        if not _is_high_signal_term(row.term):
            continue
        metric_rows.append(
            {
                "metric_id": row.metric_id,
                "metric_label": row.metric_label,
                "context_term": row.term,
                "topic": infer_topic(row.term),
                "occurrences": row.occurrences,
                "deck_count": len(row.decks),
                "sample_context_1": row.sample_snippets[0] if len(row.sample_snippets) > 0 else "",
                "sample_context_2": row.sample_snippets[1] if len(row.sample_snippets) > 1 else "",
                "sample_context_3": row.sample_snippets[2] if len(row.sample_snippets) > 2 else "",
            }
        )

    metric_rows.sort(
        key=lambda r: (
            str(r["metric_id"]),
            -int(r["occurrences"]),
            -int(r["deck_count"]),
            str(r["context_term"]).lower(),
        ),
    )

    return ranked_contexts, metric_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
