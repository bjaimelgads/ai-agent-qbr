"""SQLite -> Postgres migration helpers for QBR intelligence DB."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from sqlalchemy import JSON, Integer, create_engine, select, text
from sqlalchemy.engine import Engine

from qbr_intelligence.db.models import Base


@dataclass(frozen=True)
class MigrationResult:
    table_counts: dict[str, int]
    total_rows: int


def migrate_sqlite_to_postgres(
    *,
    sqlite_url: str,
    postgres_url: str,
    batch_size: int = 1000,
    create_tables: bool = True,
    truncate: bool = False,
    reset_sequences: bool = True,
    create_fts_indexes: bool = False,
) -> MigrationResult:
    src_engine = create_engine(sqlite_url)
    dst_engine = create_engine(postgres_url)

    if create_tables:
        Base.metadata.create_all(dst_engine)

    _widen_postgres_columns(dst_engine)

    if truncate:
        _truncate_all_tables(dst_engine)

    table_counts: dict[str, int] = {}
    total_rows = 0

    for table in Base.metadata.sorted_tables:
        count = _copy_table(table, src_engine, dst_engine, batch_size=batch_size)
        table_counts[table.name] = count
        total_rows += count

    if reset_sequences:
        _reset_postgres_sequences(dst_engine)

    if create_fts_indexes:
        create_postgres_fts_indexes(dst_engine)

    src_engine.dispose()
    dst_engine.dispose()

    return MigrationResult(table_counts=table_counts, total_rows=total_rows)


def create_postgres_fts_indexes(engine: Engine, *, language: str = "english") -> None:
    # Postgres cannot infer the type of a bound regconfig parameter here; inline it safely.
    safe_language = language.replace("'", "")
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                CREATE INDEX IF NOT EXISTS idx_chunks_fts
                ON chunks USING GIN (to_tsvector('{safe_language}'::regconfig, content));
                """
            )
        )


def _truncate_all_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


def _widen_postgres_columns(engine: Engine) -> None:
    statements = [
        (
            "ALTER TABLE IF EXISTS periods "
            "ALTER COLUMN period_label TYPE VARCHAR(200)"
        ),
        (
            "ALTER TABLE IF EXISTS periods "
            "ALTER COLUMN period_type TYPE VARCHAR(50)"
        ),
        (
            "ALTER TABLE IF EXISTS metrics "
            "ALTER COLUMN period_label TYPE VARCHAR(200)"
        ),
        (
            "ALTER TABLE IF EXISTS metrics "
            "ADD COLUMN IF NOT EXISTS llm_context_label TEXT"
        ),
    ]
    for sql_text in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql_text))
        except Exception as exc:
            message = str(exc).lower()
            if "cannot alter type of a column used by a view or rule" in message:
                print(
                    "[migration] Skipping blocked DDL due to dependent view/rule: "
                    f"{sql_text}"
                )
                continue
            if "does not exist" in message and "column" in message:
                print(
                    "[migration] Skipping DDL because referenced column is missing: "
                    f"{sql_text}"
                )
                continue
            raise


def _copy_table(table, src_engine: Engine, dst_engine: Engine, *, batch_size: int) -> int:
    json_columns = {
        column.name for column in table.columns if isinstance(column.type, JSON)
    }
    total = 0
    skipped_orphans = 0
    valid_metric_ids: set[int] | None = None
    if table.name == "metric_fact_embeddings":
        metrics_table = Base.metadata.tables.get("metrics")
        if metrics_table is not None:
            with src_engine.connect() as src_conn:
                metric_rows = src_conn.execute(select(metrics_table.c.id)).fetchall()
            valid_metric_ids = {int(r[0]) for r in metric_rows if r and r[0] is not None}

    with src_engine.connect() as src_conn:
        result = src_conn.execute(select(table))
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            payloads = []
            for row in rows:
                data = dict(row._mapping)
                if (
                    table.name == "metric_fact_embeddings"
                    and valid_metric_ids is not None
                ):
                    metric_id = data.get("metric_id")
                    if metric_id is None or int(metric_id) not in valid_metric_ids:
                        skipped_orphans += 1
                        continue
                for column_name in json_columns:
                    data[column_name] = _coerce_json(data.get(column_name))
                payloads.append(data)
            if payloads:
                with dst_engine.begin() as dst_conn:
                    dst_conn.execute(table.insert(), payloads)
                total += len(payloads)
    if skipped_orphans:
        print(
            "[migration] Skipped orphan metric_fact_embeddings rows: "
            f"{skipped_orphans}"
        )
    return total


def _coerce_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            decoded = bytes(value).decode("utf-8")
        except Exception:
            return value
        return _coerce_json(decoded)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _reset_postgres_sequences(engine: Engine) -> None:
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            pk_columns = list(table.primary_key.columns)
            if not pk_columns:
                continue
            pk_col = pk_columns[0]
            if not isinstance(pk_col.type, Integer):
                continue
            seq_name = conn.execute(
                text(
                    "SELECT pg_get_serial_sequence(:table, :column) AS seq_name"
                ),
                {"table": table.name, "column": pk_col.name},
            ).scalar()
            if not seq_name:
                continue
            conn.execute(
                text(
                    "SELECT setval(:seq, COALESCE((SELECT MAX(" + pk_col.name + ") FROM " + table.name + "), 1))"
                ),
                {"seq": seq_name},
            )


__all__ = ["migrate_sqlite_to_postgres", "create_postgres_fts_indexes", "MigrationResult"]
