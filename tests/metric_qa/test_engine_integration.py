from __future__ import annotations

import pytest

from qbr_intelligence.metric_qa import MetricQueryEngine


@pytest.mark.asyncio
async def test_query_engine_cpa(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    result = await engine.query("CPA for Nike in Q2 2025 in US")
    assert "CPA" in result.answer.summary_text or "Cost per Acquisition" in result.answer.summary_text
    assert result.answer.citations


@pytest.mark.asyncio
async def test_query_engine_trend(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    result = await engine.query("Past 4 quarters CTR for Adidas")
    assert result.answer.table_data is not None
    assert len(result.answer.table_data) >= 1


@pytest.mark.asyncio
async def test_query_engine_last_half(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    result = await engine.query("Unique reach last half in EMEA")
    assert "Unique Reach" in result.answer.summary_text or result.answer.citations


@pytest.mark.asyncio
async def test_query_engine_compare(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    result = await engine.query("Compare CPA Q1 vs Q2 2025 for Nike")
    assert "Change" in result.answer.summary_text or "moved" in result.answer.summary_text


@pytest.mark.asyncio
async def test_query_engine_definition(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    result = await engine.query("What is CPA?")
    assert "definition" in result.answer.summary_text.lower()
    assert not result.answer.citations
