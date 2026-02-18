#!/usr/bin/env python3
"""Sync missing metric aliases from Stage-4 normalization CSV into DB.

Dry-run by default. Use --apply to write to metric_aliases.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, text


ARROW_RE = re.compile(r"[\u2190-\u21ff]+")
SPACE_RE = re.compile(r"\s+")
TRAIL_NOISE_RE = re.compile(r"[\s:;,.|/-]+$")
REGEX_TOKENS_RE = re.compile(r"(\\[AbBdDsSwWZz]|[\[\]\(\)\{\}\^\$\*\+\?\|])")


@dataclass(frozen=True)
class AliasCandidate:
    metric_id: int
    canonical_name: str
    alias: str
    source_rows: int


def _clean_alias(value: str) -> str:
    text = (value or "").strip()
    text = ARROW_RE.sub("", text)
    text = SPACE_RE.sub(" ", text).strip()
    text = TRAIL_NOISE_RE.sub("", text).strip()
    return text


def _is_regex_like_alias(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    return bool(REGEX_TOKENS_RE.search(text))


def _variants(value: str) -> list[str]:
    raw = (value or "").strip()
    clean = _clean_alias(raw)
    out: list[str] = []
    for item in (raw, clean):
        item = item.strip()
        if item and item not in out:
            out.append(item)
    return out


def load_stage4_candidates(stage4_csv: Path) -> list[tuple[str, str]]:
    rows = list(csv.DictReader(stage4_csv.open(encoding="utf-8")))
    pairs: list[tuple[str, str]] = []
    for row in rows:
        if (row.get("normalization_status") or "").strip() != "catalog_mapped":
            continue
        canonical = (row.get("canonical_name") or "").strip()
        alias = (row.get("refined_name") or row.get("clean_name") or "").strip()
        if not canonical or not alias:
            continue
        pairs.append((canonical, alias))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync missing metric aliases from Stage-4 CSV.")
    parser.add_argument("--database-url", default="sqlite+aiosqlite:///qbr_intelligence.db")
    parser.add_argument("--stage4-csv", default="artifacts/metric_normalization_stage4.csv")
    parser.add_argument("--priority", type=int, default=10)
    parser.add_argument("--apply", action="store_true", help="Write new aliases to DB")
    args = parser.parse_args()

    stage4_csv = Path(args.stage4_csv)
    if not stage4_csv.exists():
        raise SystemExit(f"Stage4 CSV not found: {stage4_csv}")

    sync_db_url = args.database_url.replace("+aiosqlite", "").replace("+asyncpg", "")
    engine = create_engine(sync_db_url)

    pairs = load_stage4_candidates(stage4_csv)
    if not pairs:
        print("No catalog_mapped rows found in stage4 CSV.")
        return 0

    by_canonical = Counter(canonical for canonical, _ in pairs)

    with engine.begin() as conn:
        catalog_rows = conn.execute(text("SELECT id, name FROM metric_catalog")).fetchall()
        metric_id_by_name = {(name or "").strip().lower(): int(metric_id) for metric_id, name in catalog_rows}

        alias_rows = conn.execute(text("SELECT metric_id, alias FROM metric_aliases")).fetchall()
        existing_aliases_by_metric: dict[int, set[str]] = {}
        all_existing_aliases: dict[str, int] = {}
        for metric_id, alias in alias_rows:
            metric_id = int(metric_id)
            alias_norm = (alias or "").strip().lower()
            if not alias_norm:
                continue
            existing_aliases_by_metric.setdefault(metric_id, set()).add(alias_norm)
            all_existing_aliases[alias_norm] = metric_id

        # Also treat canonical names as already-covered aliases.
        for metric_id, name in catalog_rows:
            metric_id = int(metric_id)
            name_norm = (name or "").strip().lower()
            if name_norm:
                existing_aliases_by_metric.setdefault(metric_id, set()).add(name_norm)
                all_existing_aliases.setdefault(name_norm, metric_id)

        candidates: list[AliasCandidate] = []
        skipped_missing_canonical = 0
        skipped_existing = 0
        skipped_conflict = 0
        skipped_regex_like = 0

        seen_candidate_keys: set[tuple[int, str]] = set()
        for canonical, alias in pairs:
            metric_id = metric_id_by_name.get(canonical.lower())
            if not metric_id:
                skipped_missing_canonical += 1
                continue
            for variant in _variants(alias):
                alias_norm = variant.lower()
                key = (metric_id, alias_norm)
                if key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(key)

                # Keep aliases as plain text only; do not insert regex-like strings.
                if _is_regex_like_alias(variant):
                    skipped_regex_like += 1
                    continue

                if alias_norm in existing_aliases_by_metric.get(metric_id, set()):
                    skipped_existing += 1
                    continue
                existing_metric = all_existing_aliases.get(alias_norm)
                if existing_metric is not None and existing_metric != metric_id:
                    skipped_conflict += 1
                    continue

                candidates.append(
                    AliasCandidate(
                        metric_id=metric_id,
                        canonical_name=canonical,
                        alias=variant,
                        source_rows=by_canonical.get(canonical, 0),
                    )
                )

        print(f"Stage4 catalog_mapped pairs: {len(pairs)}")
        print(f"Missing canonical in DB: {skipped_missing_canonical}")
        print(f"Skipped existing aliases: {skipped_existing}")
        print(f"Skipped conflicting aliases: {skipped_conflict}")
        print(f"Skipped regex-like aliases: {skipped_regex_like}")
        print(f"New alias candidates: {len(candidates)}")

        if candidates:
            print("\nSample candidates:")
            for c in candidates[:30]:
                print(f"- [{c.metric_id}] {c.canonical_name} <= {c.alias}")

        if not args.apply:
            print("\nDry-run only. Use --apply to insert aliases.")
            return 0

        inserted = 0
        for c in candidates:
            conn.execute(
                text(
                    """
                    INSERT INTO metric_aliases(metric_id, alias, pattern, priority, unit_override)
                    VALUES (:metric_id, :alias, NULL, :priority, NULL)
                    """
                ),
                {
                    "metric_id": c.metric_id,
                    "alias": c.alias,
                    "priority": int(args.priority),
                },
            )
            inserted += 1

        print(f"\nInserted aliases: {inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
