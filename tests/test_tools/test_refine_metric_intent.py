from ai_agent_qbr.models import (
    CandidateSet,
    RefineMetricIntentArgs,
    ResolveMetricIntentArgs,
)
from ai_agent_qbr.tools.refine_metric_intent import refine_metric_intent
from ai_agent_qbr.tools.resolve_metric_intent import resolve_metric_intent
from qbr_intelligence.metric_qa import MetricQueryEngine


async def test_refine_metric_intent_period(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    base = await resolve_metric_intent(
        ResolveMetricIntentArgs(question="CPA for Nike"),
        dummy_ctx,
    )

    result = await refine_metric_intent(
        RefineMetricIntentArgs(
            question="CPA for Nike",
            intent=base.intent,
            candidates=CandidateSet(
                metric_ids=[],
                clients=[],
                regions=[],
                periods=["Q2 2025"],
            ),
        ),
        dummy_ctx,
    )

    assert result.intent.period
    assert result.intent.period[0].value == "Q2 2025"
    assert "period" not in result.unresolved_fields


async def test_refine_metric_intent_client_like(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    base = await resolve_metric_intent(
        ResolveMetricIntentArgs(question="CPA in US"),
        dummy_ctx,
    )

    result = await refine_metric_intent(
        RefineMetricIntentArgs(
            question="CPA in US",
            intent=base.intent,
            candidates=CandidateSet(
                metric_ids=[],
                clients=["brand"],
                regions=[],
                periods=[],
            ),
        ),
        dummy_ctx,
    )

    assert result.intent.client == ["Brand X"]
    assert "client" not in result.unresolved_fields


async def test_refine_metric_intent_fuzzy_candidate_typos(metric_db, dummy_ctx):
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine

    base = await resolve_metric_intent(
        ResolveMetricIntentArgs(question="show metric values"),
        dummy_ctx,
    )

    result = await refine_metric_intent(
        RefineMetricIntentArgs(
            question="show metric values",
            intent=base.intent,
            candidates=CandidateSet(
                metric_ids=["CPAA"],
                clients=["Nkie"],
                regions=["EMEAA"],
                periods=["Q22025"],
            ),
        ),
        dummy_ctx,
    )

    assert result.intent.metric_ids == ["cost_per_acquisition"]
    assert result.intent.client == ["Nike"]
    assert result.intent.region == ["EMEA"]
    assert result.intent.period
    assert result.intent.period[0].value == "Q2 2025"
