from ai_agent_qbr.models import CandidateSet, ResolveMetricIntentArgs
from ai_agent_qbr.tools.resolve_metric_intent import resolve_metric_intent
from qbr_intelligence.metric_qa import MetricQueryEngine


async def test_resolve_metric_intent(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await resolve_metric_intent(
        ResolveMetricIntentArgs(question="CPA for Nike in US Q2 2025"),
        dummy_ctx,
    )

    assert result.intent.metric_ids
    assert result.intent.metric_ids[0] == "cost_per_acquisition"
    assert result.intent.client == ["Nike"]
    assert result.intent.region == ["US"]
    assert result.intent.period
    assert result.intent.period[0].value == "Q2 2025"


async def test_resolve_metric_intent_multiple_regions(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await resolve_metric_intent(
        ResolveMetricIntentArgs(question="CTR in US and EMEA"),
        dummy_ctx,
    )

    assert sorted(result.intent.region) == ["EMEA", "US"]


async def test_resolve_metric_intent_uses_query_from_wrapper_payload(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    wrapped = (
        '{"observation":{"intent":{"metric_ids":["installs","impressions","completes","acquisitions"],'
        '"period":[{"value":"H2 2024"}]}},'
        '"query":"What is the CPA for Disney+ US H1 FY25?"}'
    )
    result = await resolve_metric_intent(
        ResolveMetricIntentArgs(question=wrapped),
        dummy_ctx,
    )

    assert result.intent.metric_ids
    assert result.intent.metric_ids[0] == "cost_per_acquisition"


async def test_resolve_metric_intent_extracts_multiple_half_periods_from_single_question(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await resolve_metric_intent(
        ResolveMetricIntentArgs(question="Provide CPA for Nike in US for H2 and H1 2024"),
        dummy_ctx,
    )

    period_values = {period.value for period in result.intent.period}
    assert "H1 2024" in period_values
    assert "H2 2024" in period_values
    assert len(result.intent.period) >= 2


async def test_resolve_metric_intent_applies_proposed_entities_for_missing_fields(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await resolve_metric_intent(
        ResolveMetricIntentArgs(
            question="Show me CPA in US",
            proposed_entities=CandidateSet(
                metric_ids=["CPA"],
                clients=["Nike"],
                regions=["US"],
                periods=["Q2 2025"],
            ),
        ),
        dummy_ctx,
    )

    assert result.intent.metric_ids == ["cost_per_acquisition"]
    assert result.intent.client == ["Nike"]
    assert result.intent.region == ["US"]
    assert result.intent.period
    assert result.intent.period[0].value == "Q2 2025"


async def test_resolve_metric_intent_proposed_periods_can_upgrade_to_multi_period(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await resolve_metric_intent(
        ResolveMetricIntentArgs(
            question="Provide CPA for Nike in US for H1 2024",
            proposed_entities=CandidateSet(
                periods=["H1 2024", "H2 2024"],
            ),
        ),
        dummy_ctx,
    )

    period_values = {period.value for period in result.intent.period}
    assert "H1 2024" in period_values
    assert "H2 2024" in period_values
    assert len(result.intent.period) >= 2


async def test_resolve_metric_intent_fuzzy_matches_typo_entities(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    result = await resolve_metric_intent(
        ResolveMetricIntentArgs(question="Show me CPAA for Nkie in U S for Q22025"),
        dummy_ctx,
    )

    assert result.intent.metric_ids == ["cost_per_acquisition"]
    assert result.intent.client == ["Nike"]
    assert result.intent.region == ["US"]
    assert result.intent.period
    assert result.intent.period[0].value == "Q2 2025"
