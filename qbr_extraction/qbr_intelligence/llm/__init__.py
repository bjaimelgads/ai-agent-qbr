"""LLM enhancement modules using DSPY."""

from qbr_intelligence.llm.modules import (
    ChartReconstructor,
    EntityExtractor,
    ExecutiveSummarizer,
    MetricNormalizer,
    QBREnhancementPipeline,
    SlideAnalyzer,
)

__all__ = [
    "SlideAnalyzer",
    "MetricNormalizer",
    "ChartReconstructor",
    "ExecutiveSummarizer",
    "EntityExtractor",
    "QBREnhancementPipeline",
]
