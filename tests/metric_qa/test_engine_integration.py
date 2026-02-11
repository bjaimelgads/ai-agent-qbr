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
            id, slide_id, raw_value, raw_context, raw_metric_type,
            metric_catalog_id, name, normalized_value, unit, category,
            extraction_confidence,
            brand, baseline_text, baseline_type, period_id, region_id, country,
            llm_context_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            106,
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


@pytest.mark.asyncio
async def test_query_engine_context_prefers_explicit_when_rag_confident_else_overall(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    db_path = metric_db.replace("sqlite+aiosqlite:///", "", 1)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO documents (id, filename, file_path, client_id, region_id, report_period, report_period_id, fiscal_year, quarter, half) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            20,
            "brandx_q2_2025.pptx",
            "https://docs.google.com/presentation/d/testdeck/edit",
            3,
            1,
            "Q2 2025",
            2,
            "FY25",
            "Q2",
            None,
        ),
    )
    cur.execute(
        "INSERT INTO slides (id, document_id, slide_number, google_slide_id) VALUES (?, ?, ?, ?)",
        (20, 20, 2, "slide-20"),
    )
    cur.executemany(
        """
        INSERT INTO metrics (
            id, slide_id, raw_value, raw_context, raw_metric_type,
            metric_catalog_id, name, normalized_value, unit, category,
            extraction_confidence,
            brand, baseline_text, baseline_type, period_id, region_id, country,
            llm_context_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                201,
                20,
                "$14.00",
                "CPA campaign slice in US Q2",
                "currency",
                1,
                "CPA",
                14.0,
                "currency",
                "cost",
                0.9,
                "Brand X",
                None,
                None,
                2,
                1,
                None,
                "Campaign",
            ),
            (
                202,
                20,
                "$13.00",
                "CPA overall in US Q2",
                "currency",
                1,
                "CPA",
                13.0,
                "currency",
                "cost",
                0.9,
                "Brand X",
                None,
                None,
                2,
                1,
                None,
                "Overall",
            ),
        ],
    )
    conn.commit()
    conn.close()

    engine = MetricQueryEngine(database_url=metric_db)

    generic = await engine.query("CPA for Brand X in Q2 2025 in US", debug=True)
    assert generic.answer.data is not None
    assert generic.answer.data[0].llm_context_label == "Overall"
    assert generic.debug is not None
    assert generic.debug.get("prioritize_overall_context") is True

    explicit = await engine.query(
        "Campaign CPA for Brand X in Q2 2025 in US",
        debug=True,
        rag_hit_chunks=[
            {
                "score": 0.92,
                "content": "Campaign performance detail for CPA in US Q2.",
                "summary": "Campaign context",
                "topics": ["campaign", "cpa"],
                "metadata": {},
            }
        ],
    )
    assert explicit.answer.data is not None
    assert explicit.answer.data[0].llm_context_label == "Campaign"
    assert explicit.debug is not None
    assert explicit.debug.get("prioritize_overall_context") is False

    weak_context = await engine.query(
        "Campaign CPA for Brand X in Q2 2025 in US",
        debug=True,
        rag_hit_chunks=[
            {
                "score": 0.10,
                "content": "Campaign mention without strong support.",
                "summary": "",
                "topics": [],
                "metadata": {},
            }
        ],
    )
    assert weak_context.answer.data is not None
    assert weak_context.answer.data[0].llm_context_label == "Overall"
    assert weak_context.debug is not None
    assert weak_context.debug.get("prioritize_overall_context") is True


@pytest.mark.asyncio
async def test_query_engine_half_period_does_not_leak_overlapping_half(metric_db, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    db_path = metric_db.replace("sqlite+aiosqlite:///", "", 1)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO periods (id, period_label, period_type, period_number, fiscal_year, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (5, "H1 2024", "half", 1, 2024, "2024-01-01", "2024-03-31"),
            (6, "H2 2024", "half", 2, 2024, "2024-04-01", "2024-09-30"),
        ],
    )
    cur.executemany(
        "INSERT INTO documents (id, filename, file_path, client_id, region_id, report_period, report_period_id, fiscal_year, quarter, half) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (30, "brandx_h1_2024.pptx", "https://docs.google.com/presentation/d/testdeck/edit", 3, 1, "H1 2024", 5, "FY24", None, "H1"),
            (31, "brandx_h2_2024.pptx", "https://docs.google.com/presentation/d/testdeck/edit", 3, 1, "H2 2024", 6, "FY24", None, "H2"),
        ],
    )
    cur.executemany(
        "INSERT INTO slides (id, document_id, slide_number, google_slide_id) VALUES (?, ?, ?, ?)",
        [(30, 30, 2, "slide-30"), (31, 31, 2, "slide-31")],
    )
    cur.executemany(
        """
        INSERT INTO metrics (
            id, slide_id, raw_value, raw_context, raw_metric_type,
            metric_catalog_id, name, normalized_value, unit, category,
            extraction_confidence,
            brand, baseline_text, baseline_type, period_id, region_id, country,
            llm_context_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                301,
                30,
                "$30.00",
                "CPA H1 2024 overall",
                "currency",
                1,
                "CPA",
                30.0,
                "currency",
                "cost",
                0.9,
                "Brand X",
                None,
                None,
                2,
                1,
                None,
                "overall",
            ),
            (
                302,
                31,
                "$45.00",
                "CPA H2 2024 overall",
                "currency",
                1,
                "CPA",
                45.0,
                "currency",
                "cost",
                0.9,
                "Brand X",
                None,
                None,
                2,
                1,
                None,
                "overall",
            ),
        ],
    )
    conn.commit()
    conn.close()

    engine = MetricQueryEngine(database_url=metric_db)
    intent = QueryIntent(
        metric_ids=["cost_per_acquisition"],
        client=["Brand X"],
        region=["US"],
        period=[PeriodSpec(type="half", value="H2 2024", start=date(2024, 4, 1), end=date(2024, 9, 30))],
        aggregation=None,
    )
    result = await engine.query(
        "CPA for Brand X in US for H2 2024",
        debug=True,
        intent_override=intent,
    )
    assert result.answer.data is not None
    periods = {row.period for row in result.answer.data if row.period}
    assert "H1 2024" not in periods
    assert "H2 2024" in periods
