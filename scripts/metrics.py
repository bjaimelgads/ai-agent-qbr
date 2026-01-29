#!/usr/bin/env python3
"""CLI for metric extraction pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
qbr_path = repo_root / "qbr_extraction"
if str(qbr_path) not in sys.path:
    sys.path.insert(0, str(qbr_path))

from qbr_intelligence.llm.modules import create_lm
from qbr_intelligence.metrics.adjudicator import LLMAdjudicator
from qbr_intelligence.metrics.export import export_debug, export_metrics
from qbr_intelligence.metrics.pipeline import MetricExtractionPipeline, PipelineConfig
from qbr_intelligence.metrics.parsing import compute_deck_hash


def _call_llm_text(prompt: str, model: str) -> str:
    lm = create_lm(model=model, temperature=0.0)
    for method_name in ("request", "complete", "__call__"):
        method = getattr(lm, method_name, None)
        if not callable(method):
            continue
        try:
            response = method(prompt)
        except Exception:
            continue
        if isinstance(response, str):
            return response
        for attr in ("text", "completion", "content"):
            value = getattr(response, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        try:
            return str(response)
        except Exception:
            continue
    return ""


def extract_command(args: argparse.Namespace) -> int:
    deck_path = Path(args.path)
    if not deck_path.exists():
        raise SystemExit(f"Deck not found: {deck_path}")

    adjudicator = None
    if args.llm:
        adjudicator = LLMAdjudicator(
            call_llm=lambda prompt: _call_llm_text(prompt, args.llm_model),
            cache_dir=args.cache_dir,
            enabled=True,
        )

    pipeline = MetricExtractionPipeline(
        config=PipelineConfig(),
        adjudicator=adjudicator,
    )
    metrics, debug = pipeline.extract_from_pptx(deck_path)
    export_metrics(args.out, metrics)
    if args.debug:
        export_debug(args.debug, debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metric extraction CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract metrics from PPTX")
    extract_parser.add_argument("path", help="Path to PPTX deck")
    extract_parser.add_argument("--out", required=True, help="Output JSON path")
    extract_parser.add_argument("--debug", help="Optional debug JSON path")
    extract_parser.add_argument("--llm", action="store_true", help="Enable LLM adjudication")
    extract_parser.add_argument(
        "--llm-model",
        default="openai/gpt-4o-mini",
        help="LLM model name for adjudication",
    )
    extract_parser.add_argument(
        "--cache-dir",
        default=str(repo_root / "extraction_output" / ".metric_adjudicator_cache"),
        help="Cache directory for adjudication",
    )
    extract_parser.set_defaults(func=extract_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
