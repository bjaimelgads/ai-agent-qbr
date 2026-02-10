from __future__ import annotations

from datetime import date
import sqlite3

import pytest

from qbr_intelligence.metric_qa import MetricQueryEngine, RagSlideRange
from qbr_intelligence.schemas.metric_qa import PeriodSpec, QueryIntent


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


@pytest.mark.asyncio
async def test_query_engine_applies_rag_slide_scope(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    db_path = metric_db.replace("sqlite+aiosqlite:///", "", 1)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO slides (id, document_id, slide_number, google_slide_id) VALUES (?, ?, ?, ?)",
        (8, 10, 8, "slide-8"),
    )
    cur.execute(
        """
        INSERT INTO metrics (
            id, document_id, slide_id, raw_value, raw_context, raw_metric_type,
            metric_catalog_id, name, normalized_value, unit, category,
            extraction_confidence, period_label, period_start, period_end,
            brand, baseline_text, baseline_type, period_id, region_id, country,
            llm_context_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            106,
            10,
            8,
            "$99.00",
            "CPA in US Q2 out-of-scope slide",
            "currency",
            1,
            "CPA",
            99.0,
            "currency",
            "cost",
            0.9,
            "Q2 2025",
            "2025-04-01",
            "2025-06-30",
            "Nike",
            None,
            None,
            2,
            1,
            None,
            "Nike US Q2 slide 8",
        ),
    )
    conn.commit()
    conn.close()

    engine = MetricQueryEngine(database_url=metric_db)
    intent, _ = await engine.resolve_intent("CPA for Nike in Q2 2025 in US")
    result = await engine.query(
        "CPA for Nike in Q2 2025 in US",
        intent_override=intent,
        rag_document_ids=[10],
        rag_slide_ranges=[RagSlideRange(document_id=10, start_slide=3, end_slide=4)],
    )

    assert result.answer.citations
    assert {citation.slide_id for citation in result.answer.citations} == {3}


@pytest.mark.asyncio
async def test_query_engine_parallel_scope_for_period_comparison(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    intent = QueryIntent(
        metric_ids=["cost_per_acquisition"],
        client=["Nike"],
        period=[
            PeriodSpec(type="quarter", value="Q1 2025", start=date(2025, 1, 1), end=date(2025, 3, 31)),
            PeriodSpec(type="quarter", value="Q2 2025", start=date(2025, 4, 1), end=date(2025, 6, 30)),
        ],
        aggregation="compare",
    )
    result = await engine.query(
        "Compare CPA Q1 vs Q2 2025 for Nike",
        debug=True,
        intent_override=intent,
    )
    assert result.debug is not None
    assert result.debug.get("parallel_scope") is True
    assert int(result.debug.get("scope_count", 0)) >= 2
