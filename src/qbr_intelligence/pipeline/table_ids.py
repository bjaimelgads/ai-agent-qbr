"""Shared utilities for stable table identifiers."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Sequence

_TABLE_UUID_NAMESPACE = uuid.UUID("b24595d0-24b6-4dbb-9f12-7fa7d0b88f19")


def normalize_table_rows(rows: Sequence[Sequence[str]]) -> list[list[str]]:
    normalized: list[list[str]] = []
    for row in rows:
        cleaned = [(" ".join((cell or "").split())).strip() for cell in row]
        if any(cell for cell in cleaned):
            normalized.append(cleaned)
    return normalized


def compute_table_uuid(*, slide_number: int | None, rows: Sequence[Sequence[str]]) -> str:
    """Compute deterministic UUID for a table from slide number + normalized rows."""
    payload = {
        "slide_number": int(slide_number) if slide_number is not None else None,
        "rows": normalize_table_rows(rows),
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(_TABLE_UUID_NAMESPACE, digest))

