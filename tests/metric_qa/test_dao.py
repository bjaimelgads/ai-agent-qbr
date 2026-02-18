from __future__ import annotations

import pytest

from qbr_intelligence.metric_qa.dao import MetricFactStore


class _FakeConn:
    def __init__(self) -> None:
        self.rollback_calls = 0
        self.execute_calls = 0

    async def execute(self, _stmt, _params):
        self.execute_calls += 1
        if self.execute_calls == 1:
            raise RuntimeError(
                "DBAPIError: (asyncpg.exceptions.InFailedSQLTransactionError) "
                "current transaction is aborted, commands ignored until end of transaction block"
            )
        return {"ok": True}

    async def rollback(self) -> None:
        self.rollback_calls += 1


@pytest.mark.asyncio
async def test_execute_with_abort_retry_rolls_back_and_retries_once() -> None:
    store = MetricFactStore(database_url="sqlite+aiosqlite:///:memory:")
    conn = _FakeConn()

    result = await store._execute_with_abort_retry(conn, "SELECT 1", {})

    assert result == {"ok": True}
    assert conn.rollback_calls == 1
    assert conn.execute_calls == 2
