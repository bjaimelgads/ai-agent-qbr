#!/usr/bin/env python3
"""Build Stage-4 metric normalization report across all 11c extraction outputs.

Stages implemented:
1) Ingest all metric instances from qbr_extraction/qbr_pipeline/output/**/11c_business_metrics_unfiltered.json
2) Deterministic name cleanup + quality scoring
3) Exact/alias mapping against metric catalog (code + DB when available)
4) Clustering for unresolved names, with status + confidence + validation flags

Outputs:
- artifacts/metric_normalization_stage4.csv
- artifacts/metric_normalization_stage4_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from sqlalchemy import create_engine, text

repo_root = Path(__file__).resolve().parents[5]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from qbr_intelligence.metrics.catalog import build_metric_catalog


NAME_TOKEN_RE = re.compile(r"[a-z0-9]+")
SPACE_RE = re.compile(r"\s+")
FOOTNOTE_RE = re.compile(r"\s*(?:[\*\u2020\u2021]+|\[\d+\]|\(\d+\))\s*$")
BAD_TRAIL_RE = re.compile(r"[\s:;,.|/-]+$")
NON_WORD_RE = re.compile(r"[^a-z0-9%/+& -]+")
FISCAL_TOKEN_RE = re.compile(r"\b(?:fy\d{2,4}|q[1-4]|h[12]|20\d{2}|\d{4})\b", re.IGNORECASE)
METRIC_HINT_RE = re.compile(
    r"\b(?:"
    r"ctr|vtr|vcr|cvr|roas|roi|cpa|cpi|cpe|cpc|cpm|cpv|"
    r"rate|lift|reach|impression|click|view|complete|spend|spending|investment|"
    r"acquisition|conversion|purchase|engagement|session|duration|frequency|"
    r"install|launch|mau|dau|sov|grps?|trps?|time spent|streaming hours|"
    r"viewability|fill rate|win rate|audience|users?"
    r")\b",
    re.IGNORECASE,
)
VALUE_FRAGMENT_RE = re.compile(
    r"(?:\$)?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*[KMBkmb])?(?:%|x|\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))?",
    re.IGNORECASE,
)
VALUE_IN_NAME_RE = re.compile(
    r"(?:[:=]\s*(?:\$)?[+-]?\d)|(?:\b\d+(?:\.\d+)?\s*(?:%|x|mins?|minutes?|hrs?|hours?|secs?|seconds?)\b)",
    re.IGNORECASE,
)
CONTEXT_NOISE_RE = re.compile(
    r"\b(?:countdown|source|overall|audience|vs|versus|benchmark|baseline|at a glance)\b",
    re.IGNORECASE,
)
UNIT_ONLY_NAME_RE = re.compile(
    r"^(?:mins?|minutes?|hrs?|hours?|secs?|seconds?|percent|percentage|count|ratio|currency|time|x|%|\$)$",
    re.IGNORECASE,
)
INLINE_LABEL_VALUE_RE = re.compile(
    r"^(?P<label>[^:=]{2,120}?)[\s]*[:=][\s]*(?P<value>(?:\$)?[+-]?\d.+)$",
    re.IGNORECASE,
)

DISAMBIGUATION_GROUPS = [
    {"ros"},
    {"rb", "roadblock", "roadblocks"},
    {"lapsed"},
    {"not", "installed"},
    {"rotational"},
    {"companion"},
    {"static"},
]

PROTECTED_METRIC_TOKENS = {
    "cpa",
    "cpe",
    "cpi",
    "cpc",
    "cpm",
    "cpv",
    "ctr",
    "vtr",
    "vcr",
    "cvr",
    "roas",
    "roi",
    "mau",
    "dau",
}


@dataclass(frozen=True)
class MetricInstance:
    deck: str
    path: str
    slide_number: int | None
    table_id: str | None
    raw_name: str
    clean_name: str
    normalized_name: str
    unit: str | None
    metric_type: str | None
    category: str | None
    raw_value: str | None
    normalized_value: float | None
    raw_context: str
    extraction_confidence: float | None


@dataclass(frozen=True)
class CatalogMatch:
    metric_id: int | None
    metric_slug: str | None
    metric_name: str
    default_unit: str | None
    method: str  # exact|alias


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def sanitize_name(name: str) -> str:
    text = " ".join((name or "").split()).strip()
    if not text:
        return ""
    text = FOOTNOTE_RE.sub("", text)
    text = BAD_TRAIL_RE.sub("", text)
    return text.strip()


def normalize_name(name: str) -> str:
    text = sanitize_name(name).lower().strip()
    text = NON_WORD_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip(" -:,.\t")
    return text


def token_set(name: str) -> set[str]:
    return set(NAME_TOKEN_RE.findall(normalize_name(name)))


def looks_metric_like(name: str) -> bool:
    return bool(METRIC_HINT_RE.search(name or ""))


def context_keywords(context: str) -> set[str]:
    words = set(NAME_TOKEN_RE.findall((context or "").lower()))
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "over",
        "under",
        "overall",
    }
    return {w for w in words if len(w) >= 4 and w not in stop}


def name_quality(name: str, *, occurrences: int, deck_count: int) -> tuple[float, list[str]]:
    reasons: list[str] = []
    clean = sanitize_name(name)
    norm = normalize_name(clean)
    tokens = NAME_TOKEN_RE.findall(norm)
    score = 0.78

    if not clean:
        reasons.append("empty_name")
        return 0.0, reasons
    if clean.isdigit() or re.fullmatch(r"[0-9.\-+% ]+", clean):
        reasons.append("numeric_only_name")
        score -= 0.55
    if FISCAL_TOKEN_RE.fullmatch(clean):
        reasons.append("fiscal_or_year_name")
        score -= 0.5
    if VALUE_IN_NAME_RE.search(clean):
        reasons.append("value_contamination")
        score -= 0.38
    if CONTEXT_NOISE_RE.search(clean):
        reasons.append("context_contamination")
        score -= 0.14
    if re.search(r"[a-z][A-Z]", name or ""):
        reasons.append("camel_glued_name")
        score -= 0.08
    if len(clean) > 48:
        reasons.append("name_too_long")
        score -= 0.24
    if len(tokens) > 8:
        reasons.append("too_many_tokens")
        score -= 0.18
    if not looks_metric_like(clean):
        reasons.append("missing_metric_hint")
        score -= 0.06
    if occurrences == 1:
        reasons.append("singleton_name")
        score -= 0.1
    if deck_count == 1:
        reasons.append("single_deck_name")
        score -= 0.06

    return max(0.0, min(1.0, score)), reasons


def similarity_score(
    name_a: str,
    name_b: str,
    *,
    units_a: set[str],
    units_b: set[str],
    context_a: set[str],
    context_b: set[str],
) -> float:
    na = normalize_name(name_a)
    nb = normalize_name(name_b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    seq = SequenceMatcher(a=na, b=nb).ratio()
    ta = token_set(name_a)
    tb = token_set(name_b)
    jacc = len(ta & tb) / max(1, len(ta | tb))

    acronym_bonus = 0.0
    if len(ta) == 1 and len(tb) == 1 and next(iter(ta))[:3] == next(iter(tb))[:3]:
        acronym_bonus = 0.05

    unit_bonus = 0.0
    if units_a and units_b and units_a.intersection(units_b):
        unit_bonus = 0.04

    ctx_bonus = 0.0
    if context_a and context_b:
        overlap = len(context_a & context_b) / max(1, len(context_a | context_b))
        ctx_bonus = min(0.05, overlap * 0.08)

    hint_bonus = 0.0
    if looks_metric_like(name_a) and looks_metric_like(name_b):
        hint_bonus = 0.03

    conflict_penalty = 0.0
    present_a = {idx for idx, g in enumerate(DISAMBIGUATION_GROUPS) if ta & g}
    present_b = {idx for idx, g in enumerate(DISAMBIGUATION_GROUPS) if tb & g}
    if present_a != present_b and (present_a or present_b):
        conflict_penalty = 0.22

    score = (
        (0.55 * seq)
        + (0.45 * jacc)
        + acronym_bonus
        + unit_bonus
        + ctx_bonus
        + hint_bonus
        - conflict_penalty
    )
    return max(0.0, min(1.0, score))


def choose_canonical(names: list[str], occurrences: dict[str, int]) -> str:
    def rank_key(name: str) -> tuple:
        n = sanitize_name(name)
        return (-occurrences.get(name, 0), 0 if looks_metric_like(n) else 1, len(n), n.lower())

    return sorted(names, key=rank_key)[0]


def auto_fix_allowed(alias_name: str, canonical_name: str, score: float) -> bool:
    if score < 0.93:
        return False
    a = token_set(alias_name)
    c = token_set(canonical_name)
    if len(a) > 6 or len(c) > 6:
        return False
    if len(sanitize_name(alias_name)) > 40 or len(sanitize_name(canonical_name)) > 40:
        return False
    if (a & PROTECTED_METRIC_TOKENS) != (c & PROTECTED_METRIC_TOKENS):
        return False
    return True


def is_unit_only_name(name: str) -> bool:
    return bool(UNIT_ONLY_NAME_RE.fullmatch((sanitize_name(name) or "").strip()))


def strip_value_noise(name: str) -> str:
    text = sanitize_name(name)
    if not text:
        return ""
    text = re.sub(
        r"[:=]\s*(?:\$)?[+-]?\d[\d,]*(?:\.\d+)?(?:\s*[KMBkmb])?(?:%|x|\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))?$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:\$)?[+-]?\d[\d,]*(?:\.\d+)?(?:\s*[KMBkmb])?(?:%|x|\s*(?:mins?|minutes?|hrs?|hours?|secs?|seconds?))\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = SPACE_RE.sub(" ", text).strip(" -:|,.;")
    return sanitize_name(text)


def _candidate_score(
    *,
    candidate: str,
    inst: MetricInstance,
    strategy: str,
    occurrences: dict[str, int],
    deck_count_by_name: dict[str, int],
    exact_map: dict[str, CatalogMatch],
    alias_map: dict[str, CatalogMatch],
) -> float:
    occ = occurrences.get(candidate, 1)
    decks = deck_count_by_name.get(candidate, 1)
    oq, q_reasons = name_quality(candidate, occurrences=occ, deck_count=decks)
    tokens = NAME_TOKEN_RE.findall(normalize_name(candidate))

    score = oq
    if strategy != "original":
        score += 0.03
    if strategy.startswith("context_"):
        score += 0.05
    if strategy == "strip_value_noise":
        score += 0.04

    if occ >= 3:
        score += 0.05
    if occ >= 8:
        score += 0.04
    if decks >= 2:
        score += 0.04

    norm = normalize_name(candidate)
    if norm in exact_map:
        score += 0.25
    elif norm in alias_map:
        score += 0.18

    # Penalize non-KPI short acronyms/titles (e.g., BAU) unless catalog mapped.
    if norm not in exact_map and norm not in alias_map:
        if not looks_metric_like(candidate):
            score -= 0.22
        if tokens and len(tokens) <= 2 and all(len(t) <= 3 for t in tokens):
            score -= 0.22
        if strategy == "context_title" and not looks_metric_like(candidate):
            score -= 0.12

    ctx_words = context_keywords(inst.raw_context)
    cand_words = set(NAME_TOKEN_RE.findall(norm))
    if cand_words and ctx_words:
        overlap = len(cand_words & ctx_words) / max(1, len(cand_words))
        score += min(0.08, overlap * 0.12)

    if is_unit_only_name(candidate):
        score -= 0.6
    if "value_contamination" in q_reasons:
        score -= 0.2
    if "context_contamination" in q_reasons:
        score -= 0.1

    return max(0.0, min(1.0, score))


def _extract_context_candidates(inst: MetricInstance) -> list[tuple[str, str, str]]:
    """Return candidate metric names from raw context.

    tuple: (candidate_name, strategy, context_snippet)
    """
    out: list[tuple[str, str, str]] = []
    context = inst.raw_context or ""
    if not context.strip():
        return out

    lines = [line.strip(" -\t") for line in context.splitlines() if line.strip()]
    if not lines:
        return out
    raw_value = (inst.raw_value or "").strip()

    # 1) Same-line and previous-line around value.
    if raw_value:
        for idx, line in enumerate(lines):
            if raw_value in line:
                left = line.split(raw_value, 1)[0].strip(" -:|,.;")
                left = strip_value_noise(left)
                if left:
                    out.append((left, "context_inline_value", line))
                if idx > 0:
                    prev = strip_value_noise(lines[idx - 1].strip(" -:|,.;"))
                    if prev:
                        out.append((prev, "context_prev_line", lines[idx - 1]))

    # 2) Inline label:value patterns.
    for line in lines:
        m = INLINE_LABEL_VALUE_RE.match(line)
        if not m:
            continue
        label = strip_value_noise(m.group("label"))
        if label:
            out.append((label, "context_inline_label", line))

    # 3) Short KPI-like lines.
    for line in lines:
        candidate = strip_value_noise(line)
        if not candidate:
            continue
        tokens = NAME_TOKEN_RE.findall(normalize_name(candidate))
        if 1 <= len(tokens) <= 7 and looks_metric_like(candidate):
            out.append((candidate, "context_kpi_line", line))

    # 4) Use first line as title-like fallback.
    title_like = strip_value_noise(lines[0])
    if title_like and len(NAME_TOKEN_RE.findall(normalize_name(title_like))) <= 8:
        out.append((title_like, "context_title", lines[0]))

    # Dedup preserving order.
    seen: set[str] = set()
    deduped: list[tuple[str, str, str]] = []
    for cand, strat, snippet in out:
        key = normalize_name(cand)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append((sanitize_name(cand), strat, snippet))
    return deduped


def iterative_refine_names(
    instances: list[MetricInstance],
    *,
    exact_map: dict[str, CatalogMatch],
    alias_map: dict[str, CatalogMatch],
    max_iterations: int,
) -> tuple[list[str], list[str], list[int], list[str]]:
    """Refine metric names with multi-pass context-aware strategy selection.

    Returns arrays aligned to instances:
    - refined_name
    - refinement_strategy
    - refinement_iterations
    - refinement_context_snippet
    """
    current_names = [inst.clean_name for inst in instances]
    current_strategy = ["original" for _ in instances]
    current_iters = [0 for _ in instances]
    current_context = ["" for _ in instances]
    current_scores = [0.0 for _ in instances]

    # Seed initial scores.
    seed_occ = Counter(current_names)
    seed_decks: dict[str, set[str]] = defaultdict(set)
    for inst, name in zip(instances, current_names):
        seed_decks[name].add(inst.deck)
    seed_deck_count = {k: len(v) for k, v in seed_decks.items()}
    for i, inst in enumerate(instances):
        current_scores[i] = _candidate_score(
            candidate=current_names[i],
            inst=inst,
            strategy="original",
            occurrences=dict(seed_occ),
            deck_count_by_name=seed_deck_count,
            exact_map=exact_map,
            alias_map=alias_map,
        )

    for iteration in range(1, max_iterations + 1):
        occ = Counter(current_names)
        decks_by_name: dict[str, set[str]] = defaultdict(set)
        for inst, name in zip(instances, current_names):
            decks_by_name[name].add(inst.deck)
        deck_count_by_name = {k: len(v) for k, v in decks_by_name.items()}

        changes = 0
        for i, inst in enumerate(instances):
            # Skip already catalog-mapped names; they are strong anchors.
            if normalize_name(current_names[i]) in exact_map or normalize_name(current_names[i]) in alias_map:
                continue

            candidates: list[tuple[str, str, str]] = []
            candidates.append((current_names[i], current_strategy[i], current_context[i]))

            stripped = strip_value_noise(current_names[i])
            if stripped and stripped != current_names[i]:
                candidates.append((stripped, "strip_value_noise", current_names[i]))

            candidates.extend(_extract_context_candidates(inst))

            best_name = current_names[i]
            best_score = current_scores[i]
            best_strategy = current_strategy[i]
            best_context = current_context[i]
            for cand, strat, snippet in candidates:
                cand = sanitize_name(cand)
                if not cand:
                    continue
                cand_score = _candidate_score(
                    candidate=cand,
                    inst=inst,
                    strategy=strat,
                    occurrences=dict(occ),
                    deck_count_by_name=deck_count_by_name,
                    exact_map=exact_map,
                    alias_map=alias_map,
                )
                cand_norm = normalize_name(cand)
                is_catalog = cand_norm in exact_map or cand_norm in alias_map
                # For non-catalog candidates, require KPI-like signal to replace current name.
                if not is_catalog and not looks_metric_like(cand):
                    continue
                if cand_score > best_score + 0.015:
                    best_name = cand
                    best_score = cand_score
                    best_strategy = strat
                    best_context = snippet

            if best_name != current_names[i]:
                current_names[i] = best_name
                current_scores[i] = best_score
                current_strategy[i] = best_strategy
                current_iters[i] = iteration
                current_context[i] = best_context
                changes += 1

        if changes == 0:
            break

    return current_names, current_strategy, current_iters, current_context


def load_instances(root: Path, file_name: str) -> list[MetricInstance]:
    instances: list[MetricInstance] = []
    for path in sorted(root.glob(f"**/{file_name}")):
        deck = path.parent.name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        rel = str(path.relative_to(root))
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("name") or "").strip()
            clean = sanitize_name(raw_name)
            norm = normalize_name(clean)
            if not clean:
                continue
            slide_number = row.get("slide_number")
            try:
                slide_number = int(slide_number) if slide_number is not None else None
            except Exception:
                slide_number = None
            norm_value = row.get("normalized_value")
            try:
                norm_value = float(norm_value) if norm_value is not None else None
            except Exception:
                norm_value = None
            ext_conf = row.get("extraction_confidence")
            try:
                ext_conf = float(ext_conf) if ext_conf is not None else None
            except Exception:
                ext_conf = None
            instances.append(
                MetricInstance(
                    deck=deck,
                    path=rel,
                    slide_number=slide_number,
                    table_id=(row.get("table_id") or None),
                    raw_name=raw_name,
                    clean_name=clean,
                    normalized_name=norm,
                    unit=(row.get("unit") or None),
                    metric_type=(row.get("metric_type") or None),
                    category=(row.get("category") or None),
                    raw_value=(row.get("raw_value") or None),
                    normalized_value=norm_value,
                    raw_context=str(row.get("raw_context") or ""),
                    extraction_confidence=ext_conf,
                )
            )
    return instances


def load_catalog(database_url: str) -> tuple[dict[str, CatalogMatch], dict[str, CatalogMatch]]:
    exact_map: dict[str, CatalogMatch] = {}
    alias_map: dict[str, CatalogMatch] = {}

    # code catalog
    for entry in build_metric_catalog():
        canonical = CatalogMatch(
            metric_id=None,
            metric_slug=entry.metric_id,
            metric_name=entry.name,
            default_unit=entry.expected_unit,
            method="exact",
        )
        exact_map.setdefault(normalize_name(entry.name), canonical)
        for alias in entry.aliases:
            norm_alias = normalize_name(alias.alias)
            if not norm_alias:
                continue
            alias_map.setdefault(
                norm_alias,
                CatalogMatch(
                    metric_id=None,
                    metric_slug=entry.metric_id,
                    metric_name=entry.name,
                    default_unit=alias.unit_override or entry.expected_unit,
                    method="alias",
                ),
            )

    # db catalog (override/extend)
    sync_url = database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    try:
        engine = create_engine(sync_url)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT mc.id, mc.slug, mc.name, mc.default_unit, ma.alias
                    FROM metric_catalog mc
                    LEFT JOIN metric_aliases ma ON ma.metric_id = mc.id
                    """
                )
            )
            for metric_id, slug, name, default_unit, alias in rows:
                if name:
                    exact_map[normalize_name(name)] = CatalogMatch(
                        metric_id=int(metric_id) if metric_id is not None else None,
                        metric_slug=slug,
                        metric_name=name,
                        default_unit=default_unit,
                        method="exact",
                    )
                if alias:
                    norm_alias = normalize_name(alias)
                    if norm_alias:
                        alias_map[norm_alias] = CatalogMatch(
                            metric_id=int(metric_id) if metric_id is not None else None,
                            metric_slug=slug,
                            metric_name=name,
                            default_unit=default_unit,
                            method="alias",
                        )
    except Exception:
        pass

    return exact_map, alias_map


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage-4 metric normalization CSV")
    parser.add_argument("--root", default="qbr_extraction/qbr_pipeline/output")
    parser.add_argument("--file-name", default="11c_business_metrics_unfiltered.json")
    parser.add_argument("--database-url", default="sqlite+aiosqlite:///qbr_intelligence.db")
    parser.add_argument("--cluster-threshold", type=float, default=0.86)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--out-csv", default="artifacts/metric_normalization_stage4.csv")
    parser.add_argument("--out-summary", default="artifacts/metric_normalization_stage4_summary.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    instances = load_instances(Path(args.root), args.file_name)
    if not instances:
        print("No instances found.")
        return 0

    exact_map, alias_map = load_catalog(args.database_url)
    if args.verbose:
        print(f"Catalog exact labels: {len(exact_map)}")
        print(f"Catalog aliases: {len(alias_map)}")

    refined_names, refinement_strategy, refinement_iters, refinement_context = iterative_refine_names(
        instances,
        exact_map=exact_map,
        alias_map=alias_map,
        max_iterations=max(1, args.max_iterations),
    )
    if args.verbose:
        changed = sum(1 for i, inst in enumerate(instances) if refined_names[i] != inst.clean_name)
        print(f"Refined names changed: {changed}/{len(instances)}")

    by_name: dict[str, list[MetricInstance]] = defaultdict(list)
    for i, inst in enumerate(instances):
        by_name[refined_names[i]].append(inst)

    occurrences = {name: len(rows) for name, rows in by_name.items()}
    deck_count_by_name = {name: len({r.deck for r in rows}) for name, rows in by_name.items()}
    units_by_name = {name: {r.unit for r in rows if r.unit} for name, rows in by_name.items()}
    context_by_name = {
        name: context_keywords(" ".join(r.raw_context for r in rows[:16])) for name, rows in by_name.items()
    }

    unresolved_names = []
    resolved_by_catalog: dict[str, CatalogMatch] = {}
    for name in by_name:
        norm = normalize_name(name)
        if norm in exact_map:
            resolved_by_catalog[name] = exact_map[norm]
        elif norm in alias_map:
            resolved_by_catalog[name] = alias_map[norm]
        else:
            unresolved_names.append(name)

    unresolved_names.sort()
    uf = UnionFind(len(unresolved_names))
    for i in range(len(unresolved_names)):
        for j in range(i + 1, len(unresolved_names)):
            a = unresolved_names[i]
            b = unresolved_names[j]
            score = similarity_score(
                a,
                b,
                units_a=units_by_name.get(a, set()),
                units_b=units_by_name.get(b, set()),
                context_a=context_by_name.get(a, set()),
                context_b=context_by_name.get(b, set()),
            )
            if score >= args.cluster_threshold:
                uf.union(i, j)

    clusters: dict[int, list[str]] = defaultdict(list)
    for idx, name in enumerate(unresolved_names):
        clusters[uf.find(idx)].append(name)

    cluster_id_by_name: dict[str, int] = {}
    canonical_by_name: dict[str, str] = {}
    sim_to_canonical: dict[str, float] = {}
    cluster_unit_conflict: dict[int, bool] = {}
    cluster_index = 0
    for _, members in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), sorted(kv[1])[0])):
        cluster_index += 1
        canonical = choose_canonical(members, occurrences)
        units = set()
        for name in members:
            units.update(units_by_name.get(name, set()))
        cluster_unit_conflict[cluster_index] = len(units) > 1
        for name in members:
            cluster_id_by_name[name] = cluster_index
            canonical_by_name[name] = canonical
            sim_to_canonical[name] = similarity_score(
                canonical,
                name,
                units_a=units_by_name.get(canonical, set()),
                units_b=units_by_name.get(name, set()),
                context_a=context_by_name.get(canonical, set()),
                context_b=context_by_name.get(name, set()),
            )

    rows: list[dict] = []
    status_counts: Counter[str] = Counter()
    validation_counts: Counter[str] = Counter()
    auto_cluster_rows = 0
    auto_cluster_sim_total = 0.0

    for idx, inst in enumerate(instances):
        working_name = refined_names[idx]
        oq, q_reasons = name_quality(
            working_name,
            occurrences=occurrences.get(working_name, 1),
            deck_count=deck_count_by_name.get(working_name, 1),
        )
        norm = normalize_name(working_name)
        validation_reasons = list(q_reasons)

        catalog = resolved_by_catalog.get(working_name)
        cluster_id = cluster_id_by_name.get(working_name)
        cluster_canonical = canonical_by_name.get(working_name)
        cluster_sim = sim_to_canonical.get(working_name)

        status = "needs_review"
        method = ""
        canonical_name = working_name
        catalog_id = None
        catalog_slug = None
        norm_conf = 0.0

        if "numeric_only_name" in q_reasons or "fiscal_or_year_name" in q_reasons:
            status = "invalid_name"
            method = "quality_gate"
            norm_conf = max(0.0, oq - 0.12)
        elif catalog is not None:
            canonical_name = catalog.metric_name
            catalog_id = catalog.metric_id
            catalog_slug = catalog.metric_slug
            method = f"catalog_{catalog.method}"
            status = "catalog_mapped"
            base = 0.98 if catalog.method == "exact" else 0.94
            norm_conf = max(0.0, min(1.0, base * (0.85 + 0.15 * oq)))
            if catalog.default_unit and inst.unit and catalog.default_unit != inst.unit:
                validation_reasons.append("unit_mismatch_catalog")
                norm_conf -= 0.08
                status = "needs_review"
        elif cluster_id is not None and cluster_canonical is not None:
            canonical_name = cluster_canonical
            method = "cluster"
            sim = cluster_sim or 0.0
            blocked_reasons = {
                "value_contamination",
                "context_contamination",
                "numeric_only_name",
                "fiscal_or_year_name",
                "name_too_long",
                "too_many_tokens",
                "camel_glued_name",
            }
            is_blocked = any(reason in blocked_reasons for reason in q_reasons)
            auto = (
                auto_fix_allowed(working_name, cluster_canonical, sim)
                and oq >= 0.7
                and sim >= 0.95
                and occurrences.get(working_name, 0) >= 2
                and deck_count_by_name.get(working_name, 1) >= 2
                and not is_blocked
                and not cluster_unit_conflict.get(cluster_id, False)
            )
            status = "cluster_auto" if auto else "cluster_review"
            norm_conf = max(0.0, min(1.0, (0.74 * sim) + (0.26 * oq)))
            if auto:
                norm_conf = min(1.0, norm_conf + 0.04)
                auto_cluster_rows += 1
                auto_cluster_sim_total += sim
            if is_blocked:
                norm_conf = min(norm_conf, 0.55)
            if (cluster_sim or 0.0) < 0.9:
                validation_reasons.append("low_cluster_similarity")
            if cluster_unit_conflict.get(cluster_id, False):
                validation_reasons.append("cluster_unit_conflict")
            if status == "cluster_review":
                validation_reasons.append("cluster_requires_review")
        else:
            status = "needs_review"
            method = "unresolved"
            norm_conf = max(0.0, oq - 0.22)
            validation_reasons.append("unresolved_name")

        if "value_contamination" in q_reasons:
            norm_conf = min(norm_conf, 0.5)
        if "value_contamination" in q_reasons and "singleton_name" in q_reasons:
            norm_conf = min(norm_conf, 0.45)

        ext_conf = inst.extraction_confidence if inst.extraction_confidence is not None else 0.75
        combined_conf = max(0.0, min(1.0, (0.62 * norm_conf) + (0.38 * ext_conf)))

        if status == "invalid_name":
            validation_status = "fail"
        elif combined_conf < 0.6:
            validation_status = "fail"
            validation_reasons.append("low_combined_confidence")
        elif combined_conf < 0.8 or validation_reasons:
            validation_status = "warn"
        else:
            validation_status = "pass"

        status_counts[status] += 1
        validation_counts[validation_status] += 1

        context_sample = " ".join(inst.raw_context.split())[:240]
        rows.append(
            {
                "deck": inst.deck,
                "source_path": inst.path,
                "slide_number": inst.slide_number,
                "table_id": inst.table_id or "",
                "is_table_metric": str(bool(inst.table_id)).lower(),
                "raw_name": inst.raw_name,
                "clean_name": inst.clean_name,
                "refined_name": working_name,
                "refinement_changed": str(working_name != inst.clean_name).lower(),
                "refinement_strategy": refinement_strategy[idx],
                "refinement_iteration": refinement_iters[idx],
                "refinement_context_sample": " ".join((refinement_context[idx] or "").split())[:140],
                "normalized_name": norm,
                "canonical_name": canonical_name,
                "catalog_metric_id": catalog_id if catalog_id is not None else "",
                "catalog_metric_slug": catalog_slug or "",
                "normalization_method": method,
                "normalization_status": status,
                "name_occurrences": occurrences.get(working_name, 0),
                "name_quality": round(oq, 4),
                "cluster_id": cluster_id if cluster_id is not None else "",
                "cluster_similarity": round(cluster_sim, 4) if cluster_sim is not None else "",
                "cluster_unit_conflict": str(
                    cluster_unit_conflict.get(cluster_id, False) if cluster_id is not None else False
                ).lower(),
                "normalization_confidence": round(norm_conf, 4),
                "extraction_confidence": round(ext_conf, 4),
                "combined_confidence": round(combined_conf, 4),
                "validation_status": validation_status,
                "validation_reasons": ";".join(sorted(set(validation_reasons))),
                "unit": inst.unit or "",
                "metric_type": inst.metric_type or "",
                "category": inst.category or "",
                "raw_value": inst.raw_value or "",
                "normalized_value": inst.normalized_value if inst.normalized_value is not None else "",
                "raw_context_sample": context_sample,
            }
        )

    rows.sort(
        key=lambda r: (
            r["normalization_status"],
            r["validation_status"],
            float(r["combined_confidence"]),
            r["deck"],
            int(r["slide_number"]) if str(r["slide_number"]).isdigit() else 10**9,
            r["refined_name"].lower(),
        )
    )

    # Group reviewable rows to support "review together and build correct metric".
    review_group_counts: Counter[str] = Counter()
    for row in rows:
        if row["normalization_status"] not in {"cluster_review", "needs_review", "invalid_name"}:
            continue
        group_key = (
            f"cluster:{row['cluster_id']}"
            if row["cluster_id"]
            else f"name:{normalize_name(row['refined_name'])}"
        )
        review_group_counts[group_key] += 1
    for row in rows:
        if row["normalization_status"] in {"cluster_review", "needs_review", "invalid_name"}:
            group_key = (
                f"cluster:{row['cluster_id']}"
                if row["cluster_id"]
                else f"name:{normalize_name(row['refined_name'])}"
            )
            row["review_group_key"] = group_key
            row["review_group_size"] = review_group_counts[group_key]
        else:
            row["review_group_key"] = ""
            row["review_group_size"] = ""

    write_csv(
        Path(args.out_csv),
        rows,
        [
            "deck",
            "source_path",
            "slide_number",
            "table_id",
            "is_table_metric",
            "raw_name",
            "clean_name",
            "refined_name",
            "refinement_changed",
            "refinement_strategy",
            "refinement_iteration",
            "refinement_context_sample",
            "normalized_name",
            "canonical_name",
            "catalog_metric_id",
            "catalog_metric_slug",
            "normalization_method",
            "normalization_status",
            "name_occurrences",
            "name_quality",
            "cluster_id",
            "cluster_similarity",
            "cluster_unit_conflict",
            "normalization_confidence",
            "extraction_confidence",
            "combined_confidence",
            "validation_status",
            "validation_reasons",
            "review_group_key",
            "review_group_size",
            "unit",
            "metric_type",
            "category",
            "raw_value",
            "normalized_value",
            "raw_context_sample",
        ],
    )

    unresolved_unique = len({r["clean_name"] for r in rows if r["normalization_status"] in {"cluster_review", "needs_review", "invalid_name"}})
    avg_auto_sim = (auto_cluster_sim_total / auto_cluster_rows) if auto_cluster_rows else 0.0
    total_clusters = len(set(cluster_id_by_name.values()))
    unit_conflict_clusters = sum(1 for _, v in cluster_unit_conflict.items() if v)
    conflict_ratio = (unit_conflict_clusters / total_clusters) if total_clusters else 0.0
    clustering_quality = "good"
    if avg_auto_sim < 0.93 or conflict_ratio > 0.18:
        clustering_quality = "needs_review"
    if avg_auto_sim < 0.88 or conflict_ratio > 0.3:
        clustering_quality = "poor"

    summary = {
        "instances_scanned": len(instances),
        "unique_metric_names": len(by_name),
        "catalog_mapped_unique": len({n for n in by_name if n in resolved_by_catalog}),
        "unresolved_unique_names": unresolved_unique,
        "clusters_total": total_clusters,
        "clusters_with_unit_conflict": unit_conflict_clusters,
        "cluster_unit_conflict_ratio": round(conflict_ratio, 4),
        "auto_cluster_rows": auto_cluster_rows,
        "avg_similarity_auto_cluster_rows": round(avg_auto_sim, 4),
        "clustering_quality": clustering_quality,
        "normalization_status_counts": dict(status_counts),
        "validation_status_counts": dict(validation_counts),
        "output_csv": str(Path(args.out_csv)),
    }
    Path(args.out_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Instances scanned: {summary['instances_scanned']}")
    print(f"Unique metric names: {summary['unique_metric_names']}")
    print(f"Clusters: {summary['clusters_total']} (unit_conflict={summary['clusters_with_unit_conflict']})")
    print(f"Clustering quality: {summary['clustering_quality']}")
    print(f"Normalization status: {summary['normalization_status_counts']}")
    print(f"Validation status: {summary['validation_status_counts']}")
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
