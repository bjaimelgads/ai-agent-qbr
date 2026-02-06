"""In-memory memory store for session continuity."""

from __future__ import annotations

import secrets
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from ai_agent_qbr.domain.models import MemoryInteraction


class InMemoryMemoryStore:
    """Store memory interactions in process memory."""

    def __init__(self, *, max_turns: int, retrieval_turns: int) -> None:
        self._max_turns = max_turns
        self._retrieval_turns = retrieval_turns
        self._interactions: dict[tuple[str, str, str], list[MemoryInteraction]] = defaultdict(list)

    async def start_session(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        key = (tenant_id, user_id, session_id)
        return {
            "conscious": self._serialize_interactions(self._interactions[key], limit=self._max_turns),
            "token_estimate": 0,
        }

    async def ingest_interaction(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        user_prompt: str,
        agent_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = (tenant_id, user_id, session_id)
        self._interactions[key].append(
            MemoryInteraction(
                user_prompt=user_prompt,
                agent_response=agent_response,
                created_at=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
        )
        if self._max_turns and len(self._interactions[key]) > self._max_turns:
            overflow = len(self._interactions[key]) - self._max_turns
            del self._interactions[key][:overflow]
        return {"id": secrets.token_hex(8)}

    def hydrate_interactions(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        interactions: list[MemoryInteraction],
    ) -> None:
        """Seed the in-memory store with existing interactions (no external side effects)."""
        if not interactions:
            return
        key = (tenant_id, user_id, session_id)
        self._interactions[key].extend(interactions)
        if self._max_turns and len(self._interactions[key]) > self._max_turns:
            overflow = len(self._interactions[key]) - self._max_turns
            del self._interactions[key][:overflow]

    def get_interactions(self, *, tenant_id: str, user_id: str, session_id: str) -> list[MemoryInteraction]:
        """Helper for tests to inspect stored interactions."""
        return list(self._interactions.get((tenant_id, user_id, session_id), []))

    def clear_session_by_id(self, *, session_id: str) -> None:
        """Remove all interactions for a session across tenant/user keys."""
        keys_to_remove = [key for key in self._interactions if key[2] == session_id]
        for key in keys_to_remove:
            del self._interactions[key]

    @staticmethod
    def _serialize_interactions(
        interactions: list[MemoryInteraction],
        *,
        limit: int,
    ) -> list[dict[str, str]]:
        if not interactions:
            return []
        slice_items = interactions[-limit:] if limit else interactions
        return [
            {
                "user": item.user_prompt,
                "assistant": item.agent_response,
                "metadata": item.metadata,
            }
            for item in slice_items
        ]

    def get_last_metric_intent(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        interactions = self._interactions.get((tenant_id, user_id, session_id), [])
        for item in reversed(interactions):
            intent = item.metadata.get("metric_intent") if item.metadata else None
            if isinstance(intent, dict):
                return intent
        return None
