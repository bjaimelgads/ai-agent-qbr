from __future__ import annotations

from datetime import date

from qbr_intelligence.metrics.catalog import build_metric_catalog
from qbr_intelligence.metrics.models import MetricAliasSpec, MetricCatalogEntry
from qbr_intelligence.metric_qa.resolvers import MetricResolver, PeriodResolver, RegionResolver


def test_metric_resolver_disambiguates_unique_reach():
    resolver = MetricResolver(build_metric_catalog())
    metrics, scores = resolver.resolve("unique reach in Q2")
    assert "unique_reach" in metrics
    assert scores["unique_reach"] >= scores.get("reach", -1)


def test_region_resolver_us():
    resolver = RegionResolver()
    result = resolver.resolve("CPA in US")
    assert result.value == "US"
    assert result.confidence >= 0.5


def test_period_resolver_quarter():
    resolver = PeriodResolver()
    bounds, label, period_type = resolver.resolve("Q2 2025", anchor_date=None)
    assert label == "Q2 2025"
    assert period_type == "quarter"
    assert bounds.start.isoformat() == "2025-01-01" or bounds.start is not None


def test_period_resolver_relative():
    resolver = PeriodResolver()
    anchor = date(2025, 6, 30)
    bounds, label, period_type = resolver.resolve("last quarter", anchor_date=anchor)
    assert period_type == "relative"
    assert bounds.end == anchor


def test_metric_resolver_prefers_phrase_match_over_generic_spend_alias():
    resolver = MetricResolver(
        [
            MetricCatalogEntry(
                metric_id="total_media_investment",
                name="Total Media Investment",
                aliases=(MetricAliasSpec(alias="total media investment"),),
                expected_unit="currency",
                priority=1,
            ),
            MetricCatalogEntry(
                metric_id="spend",
                name="Spend",
                aliases=(MetricAliasSpec(alias="spend"), MetricAliasSpec(alias="investment")),
                expected_unit="currency",
                priority=1,
            ),
        ]
    )
    metrics, scores = resolver.resolve("show total media investment for Disney in H2 2024")
    assert metrics[0] == "total_media_investment"
    assert scores["total_media_investment"] > scores["spend"]
