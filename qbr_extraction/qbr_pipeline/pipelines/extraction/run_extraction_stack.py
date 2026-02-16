#!/usr/bin/env python3
"""Run the current extraction stack end-to-end.

Steps:
1) qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_pipeline.py
2) qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/export_unfiltered_business_metrics.py
3) qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/build_metric_normalization_stage4.py
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction + 11c + stage4 normalization.")
    parser.add_argument("--db", default="sqlite+aiosqlite:///qbr_intelligence.db")
    parser.add_argument("--decks-folder", default="qbr_extraction/decks")
    parser.add_argument("--root", default="qbr_extraction/qbr_pipeline/output")
    parser.add_argument("--stage4-csv", default="artifacts/metric_normalization_stage4.csv")
    parser.add_argument(
        "--stage4-summary",
        default="artifacts/metric_normalization_stage4_summary.json",
    )
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--skip-run-pipeline", action="store_true")
    parser.add_argument("--skip-export-11c", action="store_true")
    parser.add_argument("--skip-stage4", action="store_true")
    args = parser.parse_args()

    decks_folder = Path(args.decks_folder)
    if not decks_folder.exists():
        raise SystemExit(f"Deck folder not found: {decks_folder}")

    if not args.skip_run_pipeline:
        cmd = [
            "uv",
            "run",
            "python",
            "qbr_extraction/qbr_pipeline/pipelines/extraction/run_extraction_pipeline.py",
            "--folder",
            str(decks_folder),
            "--no-llm",
            "--export-extraction",
            "--db",
            args.db,
        ]
        if args.override:
            cmd.append("--override")
        run_cmd(cmd)

    if not args.skip_export_11c:
        run_cmd(
            [
                "uv",
                "run",
                "python",
                "qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/export_unfiltered_business_metrics.py",
                "--root",
                args.root,
                "--out-name",
                "11c_business_metrics_unfiltered.json",
                "--review-threshold",
                "0.8",
            ]
        )

    if not args.skip_stage4:
        run_cmd(
            [
                "uv",
                "run",
                "python",
                "qbr_extraction/qbr_pipeline/pipelines/extraction/scripts/build_metric_normalization_stage4.py",
                "--root",
                args.root,
                "--database-url",
                args.db,
                "--max-iterations",
                str(max(1, args.max_iterations)),
                "--out-csv",
                args.stage4_csv,
                "--out-summary",
                args.stage4_summary,
            ]
        )

    print("Extraction stack complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
