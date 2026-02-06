"""LLM enhancement modules using DSPY."""

from qbr_intelligence.llm.modules import (
    ChartReconstructor,
    EntityExtractor,
    ExecutiveSummarizer,
    MetricContextExtractor,
    MetricDeduplicator,
    MetricNormalizer,
    MetricReviewer,
    QBREnhancementPipeline,
    SlideAnalyzer,
)

__all__ = [
    "SlideAnalyzer",
    "MetricDeduplicator",
    "MetricNormalizer",
    "MetricReviewer",
    "MetricContextExtractor",
    "ChartReconstructor",
    "ExecutiveSummarizer",
    "EntityExtractor",
    "QBREnhancementPipeline",
]
