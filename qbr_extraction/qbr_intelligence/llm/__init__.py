"""LLM enhancement modules using DSPY."""

from qbr_intelligence.llm.modules import (
    ChartReconstructor,
    EntityExtractor,
    ExecutiveSummarizer,
    MetricContextExtractor,
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
    "MetricContextExtractor",
    "ChartReconstructor",
    "ExecutiveSummarizer",
    "EntityExtractor",
    "QBREnhancementPipeline",
]
