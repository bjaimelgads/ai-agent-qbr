"""Metric extraction pipeline exports."""

import os

from qbr_intelligence.metrics.catalog import build_metric_catalog

_LIGHT_IMPORT = os.getenv("QBR_INTELLIGENCE_LIGHT_IMPORT") == "1"

if not _LIGHT_IMPORT:
    from qbr_intelligence.metrics.pipeline import MetricExtractionPipeline, PipelineConfig
    from qbr_intelligence.metrics.parsing import parse_pptx_deck

    __all__ = [
        "MetricExtractionPipeline",
        "PipelineConfig",
        "build_metric_catalog",
        "parse_pptx_deck",
    ]
else:
    __all__ = ["build_metric_catalog"]
