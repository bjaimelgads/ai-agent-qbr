#!/usr/bin/env python3
"""Export unfiltered business metrics from 02b_raw_content_by_box.txt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[5]
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from qbr_intelligence.pipeline.unfiltered_metrics_extractor import (
    UnfilteredMetricsExtractor,
    records_to_dict,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export unfiltered business metrics from 02b box content")
    parser.add_argument("--root", default="qbr_extraction/qbr_pipeline/output")
    parser.add_argument("--database-url", default="sqlite+aiosqlite:///qbr_intelligence.db")
    parser.add_argument("--out-name", default="11c_business_metrics_unfiltered.json")
    parser.add_argument(
        "--require-metric-hints",
        action="store_true",
        help="Apply metric-keyword label filtering",
    )
    parser.add_argument("--include-notes", action="store_true")
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.65,
        help="Confidence threshold below which metadata.review_recommended=true",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    extractor = UnfilteredMetricsExtractor(
        review_threshold=args.review_threshold,
        database_url=args.database_url,
    )

    root = Path(args.root)
    written = 0
    for box_file in sorted(root.glob("**/02b_raw_content_by_box.txt")):
        records = extractor.extract(
            box_file,
            require_metric_hint=args.require_metric_hints,
            include_notes=args.include_notes,
        )
        out_path = box_file.parent / args.out_name
        out_path.write_text(
            json.dumps(records_to_dict(records), indent=2),
            encoding="utf-8",
        )
        written += 1
        if args.verbose:
            review_count = sum(1 for r in records if (r.metadata or {}).get("review_recommended"))
            print(f"[wrote] {out_path} ({len(records)} records, review={review_count})")

    print(f"Decks processed: {written}")
    print(f"Output file name: {args.out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
