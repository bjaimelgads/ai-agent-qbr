from __future__ import annotations

import sqlite3

import pytest

from qbr_agent.application.ports import EmbeddingResult
from qbr_agent.domain.value_objects import EmbeddingVector
from qbr_intelligence.metric_qa import MetricQueryEngine


class FixedEmbeddingsProvider:
    async def embed_query(self, text: str) -> EmbeddingResult:
        del text
        return EmbeddingResult(vector=EmbeddingVector((1.0, 0.0, 0.0)), model="test-embed")


def _db_path(database_url: str) -> str:
    return database_url.replace("sqlite+aiosqlite:///", "")


@pytest.mark.asyncio
async def test_metric_qa_asks_for_clarification_on_ambiguous_semantic_scores(
    metric_db, monkeypatch
):
    monkeypatch.setenv("FISCAL_YEAR_START_MONTH", "1")
    monkeypatch.setenv("METRIC_SEMANTIC_RERANK_THRESHOLD", "5")
    monkeypatch.setenv("METRIC_SEMANTIC_MIN_SCORE", "0.9")
    monkeypatch.setenv("METRIC_SEMANTIC_SCORE_MARGIN", "0.2")

    db_path = _db_path(metric_db)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    extra_metrics = [
        (106, 10, 3, "12.5", "CPA Q2", "currency", 1, "CPA", 12.5, "currency", "cost", 0.9, "Q2 2025", "2025-04-01", "2025-06-30", "Nike", None, None, 2, 1, None, "Nike US Q2 alt"),
        (107, 10, 4, "11.0", "CPA Q1", "currency", 1, "CPA", 11.0, "currency", "cost", 0.9, "Q1 2025", "2025-01-01", "2025-03-31", "Nike", None, None, 1, 1, None, "Nike US Q1 alt"),
        (108, 10, 3, "13.0", "CPA Q2", "currency", 1, "CPA", 13.0, "currency", "cost", 0.9, "Q2 2025", "2025-04-01", "2025-06-30", "Nike", None, None, 2, 1, None, "Nike US Q2 alt 2"),
        (109, 10, 4, "9.5", "CPA Q1", "currency", 1, "CPA", 9.5, "currency", "cost", 0.9, "Q1 2025", "2025-01-01", "2025-03-31", "Nike", None, None, 1, 1, None, "Nike US Q1 alt 2"),
    ]
    cur.executemany(
        """
        INSERT INTO metrics (
            id, document_id, slide_id, raw_value, raw_context, raw_metric_type,
            metric_catalog_id, name, normalized_value, unit, category,
            extraction_confidence, period_label, period_start, period_end,
            brand, baseline_text, baseline_type, period_id, region_id, country,
            llm_context_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        extra_metrics,
    )

    embeddings = [(101,), (102,), (103,), (104,), (105,), (106,), (107,), (108,), (109,)]
    cur.executemany(
        """
        INSERT INTO metric_fact_embeddings (metric_id, embedding, embedding_model)
        VALUES (?, ?, ?)
        """,
        [(metric_id, "[1.0, 0.0, 0.0]", "test-embed") for (metric_id,) in embeddings],
    )
    conn.commit()
    conn.close()

    engine = MetricQueryEngine(
        database_url=metric_db,
        embeddings_provider=FixedEmbeddingsProvider(),
    )
    result = await engine.query("CPA for Nike in US")

    assert result.answer.summary_text.lower().startswith(
        "i found multiple plausible matches"
    )
    assert result.answer.followups
    assert len(result.answer.followups) <= 3
