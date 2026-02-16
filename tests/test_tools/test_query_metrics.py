from __future__ import annotations

from dataclasses import dataclass
import sqlite3

import pytest

from ai_agent_qbr.tools.query_metrics import query_metrics
from ai_agent_qbr.models import MetricQueryArgs
from qbr_agent.domain.entities import Chunk, Document, RetrievalResult
from qbr_agent.domain.value_objects import ChunkId, DocumentId, Score
from qbr_intelligence.metric_qa import MetricQueryEngine


@dataclass
class _FakeRepository:
    async def fetch_documents_by_ids(self, document_ids):
        ids = {doc_id.value for doc_id in document_ids}
        if 10 in ids:
            return [
                Document(
                    document_id=DocumentId(10),
                    filename="nike_q2_2025.pptx",
                    file_path="https://docs.google.com/presentation/d/testdeck/edit",
                    client_name="Nike",
                    period="Q2 2025",
                    status=None,
                    executive_summary=None,
                )
            ]
        return []


@dataclass
class _FakeSearchUseCase:
    repository: _FakeRepository

    async def execute(self, *, query, top_k, min_score, document_id=None):
        del query, top_k, min_score
        if document_id is not None and int(document_id) != 10:
            return []
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=ChunkId(1),
                    document_id=DocumentId(10),
                    content="Nike CPA context",
                    start_slide=3,
                    end_slide=4,
                    summary=None,
                    topics=None,
                    importance_score=None,
                    embedding=None,
                ),
                score=Score(0.9),
            )
        ]


class _RecordingSearchUseCase(_FakeSearchUseCase):
    def __init__(self, repository: _FakeRepository):
        super().__init__(repository=repository)
        self.calls: list[int | None] = []

    async def execute(self, *, query, top_k, min_score, document_id=None):
        self.calls.append(document_id)
        return await super().execute(
            query=query,
            top_k=top_k,
            min_score=min_score,
            document_id=document_id,
        )


@dataclass
class _OverallAwareSearchUseCase:
    repository: _FakeRepository

    async def execute(self, *, query, top_k, min_score, document_id=None):
        del query, top_k, min_score
        if document_id is not None and int(document_id) == 10:
            return [
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id=ChunkId(1),
                        document_id=DocumentId(10),
                        content="Nike CPA context",
                        start_slide=3,
                        end_slide=4,
                        summary=None,
                        topics=None,
                        importance_score=None,
                        embedding=None,
                    ),
                    score=Score(0.9),
                )
            ]
        if document_id is not None and int(document_id) == 13:
            return [
                RetrievalResult(
                    chunk=Chunk(
                        chunk_id=ChunkId(13),
                        document_id=DocumentId(13),
                        content="Nike overall context",
                        start_slide=4,
                        end_slide=4,
                        summary=None,
                        topics=None,
                        importance_score=None,
                        embedding=None,
                    ),
                    score=Score(0.75),
                )
            ]
        return []


@dataclass
class _MismatchedSlideSearchUseCase:
    repository: _FakeRepository

    async def execute(self, *, query, top_k, min_score, document_id=None):
        del query, top_k, min_score
        if document_id is not None and int(document_id) != 10:
            return []
        return [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id=ChunkId(99),
                    document_id=DocumentId(10),
                    content="Out-of-period slide scope",
                    start_slide=4,
                    end_slide=4,
                    summary=None,
                    topics=None,
                    importance_score=None,
                    embedding=None,
                ),
                score=Score(0.9),
            )
        ]


@pytest.mark.asyncio
async def test_query_metrics_tool(metric_db, dummy_ctx, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine
    dummy_ctx.tool_context["interaction_metadata"] = {
        "refined_metric_intent": {
            "metric_ids": ["cost_per_acquisition"],
            "client": ["Nike"],
            "region": ["US"],
            "period": [
                {
                    "type": "quarter",
                    "value": "Q2 2025",
                    "start": "2025-04-01",
                    "end": "2025-06-30",
                }
            ],
            "aggregation": None,
            "grouping": None,
            "limit": None,
        }
    }
    result = await query_metrics(
        MetricQueryArgs(question="CPA for Nike in Q2 2025 in US"),
        dummy_ctx,
    )
    assert result.status == "ok"
    assert result.metrics
    assert any((item.metric_name or "").lower().startswith("cost per acquisition") for item in result.metrics)


@pytest.mark.asyncio
async def test_query_metrics_tool_applies_rag_slide_scope(metric_db, dummy_ctx, monkeypatch):
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
    dummy_ctx.tool_context["metric_query_engine"] = engine
    dummy_ctx.tool_context["interaction_metadata"] = {
        "refined_metric_intent": {
            "metric_ids": ["cost_per_acquisition"],
            "client": ["Nike"],
            "region": ["US"],
            "period": [
                {
                    "type": "quarter",
                    "value": "Q2 2025",
                    "start": "2025-04-01",
                    "end": "2025-06-30",
                }
            ],
            "aggregation": None,
            "grouping": None,
            "limit": None,
        }
    }
    search_use_case = _RecordingSearchUseCase(repository=_FakeRepository())
    dummy_ctx.tool_context["qbr_search_use_case"] = search_use_case
    dummy_ctx.tool_context["retrieval_top_k"] = 5
    result = await query_metrics(
        MetricQueryArgs(question="CPA for Nike in Q2 2025 in US"),
        dummy_ctx,
    )

    assert result.status == "ok"
    assert {item.source.slide_id for item in result.metrics if item.source.slide_id is not None} == {3}
    assert search_use_case.calls
    assert all(call == 10 for call in search_use_case.calls)


@pytest.mark.asyncio
async def test_query_metrics_tool_returns_compact_rag_debug_for_planner(metric_db, dummy_ctx, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    interaction_metadata: dict = {
        "refined_metric_intent": {
            "metric_ids": ["cost_per_acquisition"],
            "client": ["Nike"],
            "region": ["US"],
            "period": [
                {
                    "type": "quarter",
                    "value": "Q2 2025",
                    "start": "2025-04-01",
                    "end": "2025-06-30",
                }
            ],
            "aggregation": None,
            "grouping": None,
            "limit": None,
        }
    }
    dummy_ctx.tool_context["metric_query_engine"] = engine
    dummy_ctx.tool_context["interaction_metadata"] = interaction_metadata
    dummy_ctx.tool_context["qbr_search_use_case"] = _RecordingSearchUseCase(repository=_FakeRepository())
    dummy_ctx.tool_context["retrieval_top_k"] = 5
    result = await query_metrics(
        MetricQueryArgs(question="CPA for Nike in Q2 2025 in US", debug=True),
        dummy_ctx,
    )

    assert result.status == "ok"
    assert result.intent.metric_ids == ["cost_per_acquisition"]
    assert isinstance(result.retrieval_debug, dict)
    assert "prefiltered_doc_ids" in result.retrieval_debug
    assert all((len(str(item.content or "")) <= 400) for item in result.retrieval_chunks)

    full_metric_answer = interaction_metadata.get("metric_answer")
    assert isinstance(full_metric_answer, dict)
    full_debug = full_metric_answer.get("debug")
    assert isinstance(full_debug, dict)
    full_rag_scope = full_debug.get("rag_scope")
    assert isinstance(full_rag_scope, dict)
    assert "retrieved_hit_chunks" in full_rag_scope
    assert "post_rerank_chunks_full" in full_rag_scope


@pytest.mark.asyncio
async def test_query_metrics_tool_falls_back_when_rag_scope_prunes_all_rows(metric_db, dummy_ctx, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine
    dummy_ctx.tool_context["interaction_metadata"] = {
        "refined_metric_intent": {
            "metric_ids": ["cost_per_acquisition"],
            "client": ["Nike"],
            "region": ["US"],
            "period": [
                {
                    "type": "quarter",
                    "value": "Q2 2025",
                    "start": "2025-04-01",
                    "end": "2025-06-30",
                }
            ],
            "aggregation": None,
            "grouping": None,
            "limit": None,
        }
    }
    dummy_ctx.tool_context["qbr_search_use_case"] = _MismatchedSlideSearchUseCase(repository=_FakeRepository())
    dummy_ctx.tool_context["retrieval_top_k"] = 5

    result = await query_metrics(
        MetricQueryArgs(question="CPA for Nike in Q2 2025 in US", debug=True),
        dummy_ctx,
    )

    assert result.status == "ok"
    assert {item.source.slide_id for item in result.metrics if item.source.slide_id is not None} == {3}
    full_metric_answer = dummy_ctx.tool_context["interaction_metadata"].get("metric_answer")
    assert isinstance(full_metric_answer, dict)
    full_debug = full_metric_answer.get("debug")
    assert isinstance(full_debug, dict)
    full_rag_scope = full_debug.get("rag_scope")
    assert isinstance(full_rag_scope, dict)
    assert full_rag_scope.get("metric_slide_filtered_hit_count") == 0


@pytest.mark.asyncio
async def test_query_metrics_tool_keeps_refined_filters_strict_when_no_rows(metric_db, dummy_ctx, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    engine = MetricQueryEngine(database_url=metric_db)
    dummy_ctx.tool_context["metric_query_engine"] = engine
    dummy_ctx.tool_context["interaction_metadata"] = {
        "refined_metric_intent": {
            "metric_ids": ["cost_per_acquisition"],
            "client": ["Nike"],
            "region": ["US"],
            "period": [
                {
                    "type": "quarter",
                    "value": "Q4 2026",
                    "start": "2026-10-01",
                    "end": "2026-12-31",
                }
            ],
            "aggregation": None,
            "grouping": None,
            "limit": None,
        }
    }
    result = await query_metrics(
        MetricQueryArgs(question="CPA for Nike in Q4 2026 in US", debug=True),
        dummy_ctx,
    )

    assert result.status == "empty"
    assert result.metrics == []


@pytest.mark.asyncio
async def test_query_metrics_tool_always_includes_overall_when_metric_exists(metric_db, dummy_ctx, monkeypatch):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    db_path = metric_db.replace("sqlite+aiosqlite:///", "", 1)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Insert an overall baseline row for the metric in a different period (Q1),
    # so strict period filtering misses it but relaxed overall fallback can include it.
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
            206,
            5,
            "$11.00",
            "CPA overall in US Q1",
            "currency",
            1,
            "CPA",
            11.0,
            "currency",
            "cost",
            0.9,
            "Nike",
            "overall",
            "overall",
            1,
            1,
            None,
            "Overall",
        ),
    )
    conn.commit()
    conn.close()

    engine = MetricQueryEngine(database_url=metric_db)
    interaction_metadata: dict = {
        "refined_metric_intent": {
            "metric_ids": ["cost_per_acquisition"],
            "client": ["Nike"],
            "region": ["US"],
            "period": [
                {
                    "type": "quarter",
                    "value": "Q2 2025",
                    "start": "2025-04-01",
                    "end": "2025-06-30",
                }
            ],
            "aggregation": None,
            "grouping": None,
            "limit": None,
        }
    }
    dummy_ctx.tool_context["metric_query_engine"] = engine
    dummy_ctx.tool_context["interaction_metadata"] = interaction_metadata
    dummy_ctx.tool_context["qbr_search_use_case"] = _OverallAwareSearchUseCase(repository=_FakeRepository())
    dummy_ctx.tool_context["retrieval_top_k"] = 5

    result = await query_metrics(
        MetricQueryArgs(question="CPA for Nike in Q2 2025 in US", debug=True),
        dummy_ctx,
    )

    assert result.status == "ok"
    assert isinstance(result.retrieval_debug, dict)
    assert result.retrieval_debug.get("overall_scope_mode") == "client_metric"
    assert (result.retrieval_debug.get("appended_overall_row_count") or 0) >= 1
    assert any(
        chunk.document_id == 13 and chunk.matched_metric_slide is False
        for chunk in result.retrieval_chunks
    )
