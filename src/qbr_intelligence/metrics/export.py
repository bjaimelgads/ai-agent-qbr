"""Export helpers for metric extraction outputs."""

from __future__ import annotations

import json
from pathlib import Path

from qbr_intelligence.metrics.models import ExtractionDebug, MetricExtraction


def export_metrics(path: str | Path, metrics: list[MetricExtraction]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [metric.to_dict() for metric in metrics]
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_debug(path: str | Path, debug: ExtractionDebug) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(debug.to_dict(), indent=2, default=str), encoding="utf-8")
