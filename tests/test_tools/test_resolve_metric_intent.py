from ai_agent_qbr.models import ResolveMetricIntentArgs
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
    assert result.intent.client == "Nike"
    assert result.intent.region == "US"
    assert result.intent.period is not None
    assert result.intent.period.value == "Q2 2025"
