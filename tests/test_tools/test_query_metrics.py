from __future__ import annotations

import pytest

from ai_agent_qbr.tools.query_metrics import query_metrics
from ai_agent_qbr.models import MetricQueryArgs
from qbr_intelligence.metric_qa import MetricQueryEngine


class DummyContext:
    def __init__(self, engine):
        self.tool_context = {
            "metric_query_engine": engine,
        }


@pytest.mark.asyncio
async def test_query_metrics_tool(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    ctx = DummyContext(engine)
    result = await query_metrics(MetricQueryArgs(question="CPA for Nike in Q2 2025 in US"), ctx)
    assert "CPA" in result.summary_text or "Cost per Acquisition" in result.summary_text
    assert result.citations
