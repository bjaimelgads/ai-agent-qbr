#!/usr/bin/env python3
"""Discover non-metric context terms around metrics from extraction outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[5]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from qbr_intelligence.pipeline.context_terms_discovery import (
    discover_metric_context_terms,
    write_csv,
)


DEFAULT_DB_URL = "sqlite+aiosqlite:///qbr_intelligence.db"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Discover important non-metric context terms near known metrics "
            "(placements, audiences, creative units, ad products)."
        )
    )
    parser.add_argument(
        "--root",
        default="qbr_extraction/qbr_pipeline/output",
        help="Root directory containing extracted deck output folders",
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DB_URL,
        help="Database URL used to enrich known metric aliases from metric_catalog/metric_aliases",
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=3,
        help="Keep only context terms with at least this many occurrences",
    )
    parser.add_argument(
        "--out-context",
        default="artifacts/metric_context_terms.csv",
        help="Output CSV with global important context terms",
    )
    parser.add_argument(
        "--out-by-metric",
        default="artifacts/metric_context_terms_by_metric.csv",
        help="Output CSV with context terms per metric",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Max number of rows to write in global context CSV",
    )
    parser.add_argument(
        "--top-n-per-metric",
        type=int,
        default=80,
        help="Max rows per metric in by-metric CSV",
    )
    return parser


def _limit_by_metric(rows: list[dict[str, object]], per_metric_limit: int) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    out: list[dict[str, object]] = []
    for row in rows:
        metric_id = str(row.get("metric_id") or "")
        counts.setdefault(metric_id, 0)
        if counts[metric_id] >= per_metric_limit:
            continue
        out.append(row)
        counts[metric_id] += 1
    return out


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root)
    out_context = Path(args.out_context)
    out_by_metric = Path(args.out_by_metric)

    contexts, by_metric = discover_metric_context_terms(
        root=root,
        database_url=args.database_url,
        min_occurrences=args.min_occurrences,
    )

    contexts = contexts[: max(args.top_n, 1)]
    by_metric = _limit_by_metric(by_metric, max(args.top_n_per_metric, 1))

    write_csv(
        out_context,
        contexts,
        fieldnames=[
            "context_term",
            "topic",
            "importance_score",
            "occurrences",
            "deck_count",
            "metric_count",
            "source_table_dimension",
            "source_narrative_window",
            "sample_context_1",
            "sample_context_2",
            "sample_context_3",
        ],
    )
    write_csv(
        out_by_metric,
        by_metric,
        fieldnames=[
            "metric_id",
            "metric_label",
            "context_term",
            "topic",
            "occurrences",
            "deck_count",
            "sample_context_1",
            "sample_context_2",
            "sample_context_3",
        ],
    )

    print(f"Scanned root: {root}")
    print(f"Global context rows written: {len(contexts)}")
    print(f"Per-metric context rows written: {len(by_metric)}")
    print(f"Output (global): {out_context}")
    print(f"Output (by metric): {out_by_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
