"""LLM enhancement modules using DSPY."""

from qbr_intelligence.llm.modules import (
    ChartReconstructor,
    EntityExtractor,
    ExecutiveSummarizer,
    MetricDeduplicator,
    MetricNormalizer,
    MetricRefiner,
    QBREnhancementPipeline,
    SlideAnalyzer,
)

__all__ = [
    "SlideAnalyzer",
    "MetricDeduplicator",
    "MetricNormalizer",
    "MetricRefiner",
    "ChartReconstructor",
    "ExecutiveSummarizer",
    "EntityExtractor",
    "QBREnhancementPipeline",
]
