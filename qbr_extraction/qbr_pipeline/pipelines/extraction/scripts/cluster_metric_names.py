#!/usr/bin/env python3
"""Cluster similar metric names across all extracted decks.

Input:
- Recursively scans ``<root>/**/11c_business_metrics_unfiltered.json``.

Outputs (CSV in ``--out-dir``):
- ``metric_name_inventory.csv``: unique metric names + quality signals.
- ``metric_name_clusters.csv``: cluster membership and similarity-to-canonical.
- ``metric_alias_suggestions.csv``: alias -> canonical mapping suggestions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


NAME_TOKEN_RE = re.compile(r"[a-z0-9]+")
FOOTNOTE_RE = re.compile(r"\s*(?:[\*\u2020\u2021]+|\[\d+\]|\(\d+\))\s*$")
BAD_TRAIL_RE = re.compile(r"[\s:;,.|/-]+$")
METRIC_HINT_RE = re.compile(
    r"\b(?:"
    r"ctr|vtr|vcr|cvr|roas|roi|cpa|cpi|cpe|cpc|cpm|cpv|"
    r"rate|lift|reach|impression|click|view|complete|spend|acquisition|conversion|purchase|"
    r"engagement|session|duration|frequency|install|launch|mau|dau|sov|grps?|trps?|"
    r"time spent|streaming hours|viewability|fill rate|win rate"
    r")\b",
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
    slide_number: int | None
    name: str
    unit: str | None
    category: str | None
    context: str


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
    return " ".join(NAME_TOKEN_RE.findall(sanitize_name(name).lower()))


def token_set(name: str) -> set[str]:
    return set(NAME_TOKEN_RE.findall(normalize_name(name)))


def looks_metric_like(name: str) -> bool:
    return bool(METRIC_HINT_RE.search(name or ""))


def context_keywords(context: str) -> set[str]:
    words = set(NAME_TOKEN_RE.findall((context or "").lower()))
    stop = {"the", "and", "for", "with", "this", "that", "from", "into", "over", "under", "overall"}
    return {w for w in words if len(w) >= 4 and w not in stop}


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
    if len(ta) == 1 and len(tb) == 1:
        if next(iter(ta))[:3] == next(iter(tb))[:3]:
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
        # Same surface form but different strategy/channel discriminator.
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
        return (
            -occurrences.get(name, 0),
            0 if looks_metric_like(n) else 1,
            len(n),
            n.lower(),
        )

    return sorted(names, key=rank_key)[0]


def auto_fix_allowed(alias_name: str, canonical_name: str, score: float) -> bool:
    if score < 0.92:
        return False
    a = token_set(alias_name)
    c = token_set(canonical_name)
    if len(a) > 6 or len(c) > 6:
        return False
    if len(sanitize_name(alias_name)) > 36 or len(sanitize_name(canonical_name)) > 36:
        return False
    a_prot = a & PROTECTED_METRIC_TOKENS
    c_prot = c & PROTECTED_METRIC_TOKENS
    if a_prot != c_prot:
        return False
    return True


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
        for row in payload:
            if not isinstance(row, dict):
                continue
            raw_name = str(row.get("name") or "").strip()
            name = sanitize_name(raw_name)
            if not name:
                continue
            instances.append(
                MetricInstance(
                    deck=deck,
                    slide_number=row.get("slide_number"),
                    name=name,
                    unit=(row.get("unit") or None),
                    category=(row.get("category") or None),
                    context=str(row.get("raw_context") or ""),
                )
            )
    return instances


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cluster similar metric names across decks")
    parser.add_argument("--root", default="qbr_extraction/qbr_pipeline/output")
    parser.add_argument("--file-name", default="11c_business_metrics_unfiltered.json")
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--cluster-threshold", type=float, default=0.86)
    parser.add_argument("--auto-fix-threshold", type=float, default=0.92)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    instances = load_instances(root, args.file_name)
    if not instances:
        print("No metric instances found.")
        return 0

    by_name: dict[str, list[MetricInstance]] = defaultdict(list)
    for inst in instances:
        by_name[inst.name].append(inst)
    unique_names = sorted(by_name.keys())

    occurrences = {name: len(rows) for name, rows in by_name.items()}
    decks_by_name = {name: sorted({r.deck for r in rows}) for name, rows in by_name.items()}
    units_by_name = {name: sorted({(r.unit or "") for r in rows if r.unit}) for name, rows in by_name.items()}
    categories_by_name = {
        name: sorted({(r.category or "") for r in rows if r.category}) for name, rows in by_name.items()
    }
    contexts_by_name = {
        name: context_keywords(" ".join(r.context for r in rows[:12])) for name, rows in by_name.items()
    }

    uf = UnionFind(len(unique_names))
    for i in range(len(unique_names)):
        for j in range(i + 1, len(unique_names)):
            a = unique_names[i]
            b = unique_names[j]
            score = similarity_score(
                a,
                b,
                units_a=set(units_by_name.get(a, [])),
                units_b=set(units_by_name.get(b, [])),
                context_a=contexts_by_name.get(a, set()),
                context_b=contexts_by_name.get(b, set()),
            )
            if score >= args.cluster_threshold:
                uf.union(i, j)

    clusters: dict[int, list[str]] = defaultdict(list)
    for idx, name in enumerate(unique_names):
        clusters[uf.find(idx)].append(name)

    cluster_rows: list[dict] = []
    alias_rows: list[dict] = []
    inventory_rows: list[dict] = []

    cluster_index = 0
    for _, names in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), sorted(kv[1])[0])):
        cluster_index += 1
        canonical = choose_canonical(names, occurrences)
        for name in sorted(names):
            score = similarity_score(
                canonical,
                name,
                units_a=set(units_by_name.get(canonical, [])),
                units_b=set(units_by_name.get(name, [])),
                context_a=contexts_by_name.get(canonical, set()),
                context_b=contexts_by_name.get(name, set()),
            )
            action = "keep"
            if name != canonical:
                action = (
                    "auto_fix"
                    if auto_fix_allowed(name, canonical, score)
                    else "review"
                )
                alias_rows.append(
                    {
                        "cluster_id": cluster_index,
                        "alias_name": name,
                        "canonical_name": canonical,
                        "similarity": f"{score:.4f}",
                        "occurrences": occurrences.get(name, 0),
                        "decks": "; ".join(decks_by_name.get(name, [])),
                        "units": "; ".join(units_by_name.get(name, [])),
                        "action": action,
                    }
                )

            cluster_rows.append(
                {
                    "cluster_id": cluster_index,
                    "canonical_name": canonical,
                    "member_name": name,
                    "member_occurrences": occurrences.get(name, 0),
                    "similarity_to_canonical": f"{score:.4f}",
                    "decks": "; ".join(decks_by_name.get(name, [])),
                    "units": "; ".join(units_by_name.get(name, [])),
                    "action": action,
                }
            )

    canonical_by_name = {row["member_name"]: row["canonical_name"] for row in cluster_rows}
    for name in unique_names:
        rows = by_name[name]
        sample_context = next((r.context for r in rows if r.context), "")
        sample_context = " ".join(sample_context.split())[:220]
        inventory_rows.append(
            {
                "metric_name": name,
                "normalized_name": normalize_name(name),
                "canonical_name": canonical_by_name.get(name, name),
                "occurrences": len(rows),
                "deck_count": len(decks_by_name.get(name, [])),
                "decks": "; ".join(decks_by_name.get(name, [])),
                "units": "; ".join(units_by_name.get(name, [])),
                "categories": "; ".join(categories_by_name.get(name, [])),
                "looks_metric_like": str(looks_metric_like(name)).lower(),
                "name_length": len(name),
                "token_count": len(token_set(name)),
                "sample_context": sample_context,
            }
        )

    inventory_rows.sort(key=lambda r: (-int(r["occurrences"]), r["metric_name"].lower()))
    cluster_rows.sort(key=lambda r: (int(r["cluster_id"]), r["member_name"].lower()))
    alias_rows.sort(key=lambda r: (r["action"], -int(r["occurrences"]), r["alias_name"].lower()))

    write_csv(
        out_dir / "metric_name_inventory.csv",
        inventory_rows,
        [
            "metric_name",
            "normalized_name",
            "canonical_name",
            "occurrences",
            "deck_count",
            "decks",
            "units",
            "categories",
            "looks_metric_like",
            "name_length",
            "token_count",
            "sample_context",
        ],
    )
    write_csv(
        out_dir / "metric_name_clusters.csv",
        cluster_rows,
        [
            "cluster_id",
            "canonical_name",
            "member_name",
            "member_occurrences",
            "similarity_to_canonical",
            "decks",
            "units",
            "action",
        ],
    )
    write_csv(
        out_dir / "metric_name_alias_suggestions.csv",
        alias_rows,
        [
            "cluster_id",
            "alias_name",
            "canonical_name",
            "similarity",
            "occurrences",
            "decks",
            "units",
            "action",
        ],
    )

    auto_fix = sum(1 for row in alias_rows if row["action"] == "auto_fix")
    review = sum(1 for row in alias_rows if row["action"] == "review")
    print(f"Instances scanned: {len(instances)}")
    print(f"Unique names: {len(unique_names)}")
    print(f"Clusters: {len(clusters)}")
    print(f"Alias suggestions: {len(alias_rows)} (auto_fix={auto_fix}, review={review})")
    print(f"Wrote: {out_dir / 'metric_name_inventory.csv'}")
    print(f"Wrote: {out_dir / 'metric_name_clusters.csv'}")
    print(f"Wrote: {out_dir / 'metric_name_alias_suggestions.csv'}")
    if args.verbose and alias_rows:
        print("\nTop alias suggestions:")
        for row in alias_rows[:20]:
            print(
                f"  [{row['action']}] {row['alias_name']} -> {row['canonical_name']} "
                f"(sim={row['similarity']}, occ={row['occurrences']})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
