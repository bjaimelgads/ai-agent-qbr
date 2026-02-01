"""Pytest configuration for ai-agent-qbr."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from ai_agent_qbr.config import Config
import sqlite3


@pytest.fixture()
def metric_db(tmp_path: Path) -> str:
    db_path = tmp_path / "metric_qa.db"
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        CREATE TABLE clients (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE regions (id INTEGER PRIMARY KEY, code TEXT, name TEXT);
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            client_id INTEGER,
            region_id INTEGER,
            report_period TEXT
        );
        CREATE TABLE periods (
            id INTEGER PRIMARY KEY,
            period_label TEXT,
            period_type TEXT,
            period_number INTEGER,
            fiscal_year INTEGER,
            start_date TEXT,
            end_date TEXT
        );
        CREATE TABLE metric_catalog (
            id INTEGER PRIMARY KEY,
            name TEXT,
            slug TEXT,
            category TEXT,
            default_unit TEXT,
            description TEXT,
            formula TEXT
        );
        CREATE TABLE metric_aliases (
            id INTEGER PRIMARY KEY,
            metric_id INTEGER,
            alias TEXT,
            pattern TEXT,
            priority INTEGER,
            unit_override TEXT
        );
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            slide_id INTEGER,
            raw_value TEXT NOT NULL,
            raw_context TEXT,
            raw_metric_type TEXT NOT NULL,
            metric_catalog_id INTEGER,
            name TEXT,
            normalized_value FLOAT,
            unit TEXT,
            category TEXT,
            extraction_confidence FLOAT,
            period_label TEXT,
            period_start TEXT,
            period_end TEXT,
            brand TEXT,
            baseline_text TEXT,
            baseline_type TEXT,
            period_id INTEGER,
            region_id INTEGER,
            country TEXT
        );
        """
    )

    cur.executemany(
        "INSERT INTO clients (id, name) VALUES (?, ?)",
        [(1, "Nike"), (2, "Adidas"), (3, "Brand X")],
    )
    cur.executemany(
        "INSERT INTO regions (id, code, name) VALUES (?, ?, ?)",
        [(1, "US", "United States"), (2, "EMEA", "EMEA")],
    )
    cur.executemany(
        "INSERT INTO documents (id, filename, client_id, region_id, report_period) VALUES (?, ?, ?, ?, ?)",
        [
            (10, "nike_q2_2025.pptx", 1, 1, "Q2 2025"),
            (11, "adidas_q2_2025.pptx", 2, 2, "Q2 2025"),
            (12, "brandx_q1_2025.pptx", 3, 1, "Q1 2025"),
        ],
    )
    cur.executemany(
        "INSERT INTO periods (id, period_label, period_type, period_number, fiscal_year, start_date, end_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "Q1 2025", "quarter", 1, 2025, "2025-01-01", "2025-03-31"),
            (2, "Q2 2025", "quarter", 2, 2025, "2025-04-01", "2025-06-30"),
            (3, "Q3 2024", "quarter", 3, 2024, "2024-07-01", "2024-09-30"),
            (4, "H2 2024", "half", 2, 2024, "2024-07-01", "2024-12-31"),
        ],
    )
    cur.executemany(
        "INSERT INTO metric_catalog (id, name, slug, category, default_unit, description, formula) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "Cost per Acquisition", "cost_per_acquisition", "cost", "currency", "Cost per acquisition definition.", "spend / acquisitions"),
            (2, "Click Through Rate", "click_through_rate", "engagement", "percent", None, None),
            (3, "Unique Reach", "unique_reach", "reach", "count", None, None),
        ],
    )
    cur.executemany(
        "INSERT INTO metric_aliases (id, metric_id, alias, pattern, priority, unit_override) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, 1, "CPA", None, 2, None),
            (2, 2, "CTR", None, 2, None),
            (3, 3, "unique reach", None, 2, None),
        ],
    )
    cur.executemany(
        """
        INSERT INTO metrics (
            id, document_id, slide_id, raw_value, raw_context, raw_metric_type,
            metric_catalog_id, name, normalized_value, unit, category,
            extraction_confidence, period_label, period_start, period_end,
            brand, baseline_text, baseline_type, period_id, region_id, country
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (101, 10, 3, "$12.00", "CPA in US Q2", "currency", 1, "CPA", 12.0, "currency", "cost", 0.9, "Q2 2025", "2025-04-01", "2025-06-30", "Nike", None, None, 2, 1, None),
            (102, 10, 4, "$10.00", "CPA in US Q1", "currency", 1, "CPA", 10.0, "currency", "cost", 0.9, "Q1 2025", "2025-01-01", "2025-03-31", "Nike", None, None, 1, 1, None),
            (103, 11, 6, "1.2%", "CTR in EMEA", "percent", 2, "CTR", 1.2, "percent", "engagement", 0.8, "Q2 2025", "2025-04-01", "2025-06-30", "Adidas", None, None, 2, 2, None),
            (104, 11, 7, "1.0%", "CTR in EMEA", "percent", 2, "CTR", 1.0, "percent", "engagement", 0.8, "Q3 2024", "2024-07-01", "2024-09-30", "Adidas", None, None, 3, 2, None),
            (105, 12, 2, "120000", "Unique reach EMEA", "count", 3, "Unique Reach", 120000, "count", "reach", 0.7, "H2 2024", "2024-07-01", "2024-12-31", "Brand X", None, None, 4, 2, None),
        ],
    )
    conn.commit()
    conn.close()

    return f"sqlite+aiosqlite:///{db_path}"


@dataclass
class DummyToolContext:
    """Mock ToolContext for testing tools without a full planner.

    Captures emit_chunk() calls and provides basic context fields.
    """

    llm_context: dict[str, Any] = field(default_factory=dict)
    tool_context: dict[str, Any] = field(default_factory=lambda: {"tenant_id": "test-tenant"})
    meta: dict[str, Any] = field(default_factory=dict)
    chunks: list[tuple[str, int, str, bool]] = field(default_factory=list)
    updates: list[object] = field(default_factory=list)

    async def pause(self, *args: Any, **kwargs: Any) -> None:
        """Not implemented in tests - raises if called unexpectedly."""
        del args, kwargs
        raise RuntimeError("Unexpected pause() call in test")

    async def emit_chunk(
        self,
        stream_id: str,
        seq: int,
        text: str,
        done: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Record emitted chunks for assertions."""
        del meta  # Not tracked in basic tests
        self.chunks.append((stream_id, seq, text, done))

    def record_status(self, update: object) -> None:
        """Record status updates for assertions."""
        self.updates.append(update)


@pytest.fixture
def dummy_ctx() -> DummyToolContext:
    """Provide a fresh DummyToolContext for each test."""
    return DummyToolContext()


@pytest.fixture
def config() -> Config:
    """Provide default Config for testing."""
    return Config()
