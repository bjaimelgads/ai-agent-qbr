"""Structured metric QA package."""

from .engine import MetricQueryEngine, MetricQueryResult, RagSlideRange
from .dao import MetricFactStore
from .intent import DeterministicIntentExtractor
from .resolvers import MetricResolver, ClientResolver, RegionResolver, PeriodResolver

__all__ = [
    "MetricQueryEngine",
    "MetricQueryResult",
    "RagSlideRange",
    "MetricFactStore",
    "DeterministicIntentExtractor",
    "MetricResolver",
    "ClientResolver",
    "RegionResolver",
    "PeriodResolver",
]
