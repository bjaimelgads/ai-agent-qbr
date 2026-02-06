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

    assert result.intent.period is not None
    assert result.intent.period.value == "Q2 2025"
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

    assert result.intent.client == "Brand X"
    assert "client" not in result.unresolved_fields
