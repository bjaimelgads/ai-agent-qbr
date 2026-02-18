"""Message repository for caching and message creation."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

from ai_agent_qbr.domain.models import ChatMessage

logger = logging.getLogger(__name__)


class MessageRepositoryInterface(ABC):
    """Interface for message repository operations."""

    @abstractmethod
    async def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        before_timestamp: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Get cached messages for a session."""

    @abstractmethod
    async def cache_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        cache_key: str,
    ) -> None:
        """Cache messages for future retrieval."""

    @abstractmethod
    async def create_chat_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        has_chart: bool = False,
        artifacts: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ChatMessage:
        """Create a ChatMessage domain object."""

    @abstractmethod
    def invalidate_cache(self, session_id: str) -> None:
        """Invalidate message cache for a session."""

    @abstractmethod
    def build_cache_key(
        self,
        session_id: str,
        limit: int,
        before_timestamp: Optional[datetime] = None,
    ) -> str:
        """Build a cache key for message queries."""


class MessageRepository(MessageRepositoryInterface):
    """In-memory message cache with TTL expiration."""

    def __init__(self, *, cache_ttl_seconds: int = 180) -> None:
        self._cache_ttl_seconds = cache_ttl_seconds
        self._message_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        logger.info("MessageRepository initialized (ttl=%ss)", cache_ttl_seconds)

    async def get_messages(
        self,
        session_id: str,
        limit: int = 50,
        before_timestamp: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        cache_key = self.build_cache_key(session_id, limit, before_timestamp)
        cached = self._message_cache.get(cache_key)
        if not cached:
            return []
        expires_at, messages = cached
        if time.monotonic() >= expires_at:
            self._message_cache.pop(cache_key, None)
            return []
        logger.debug("Retrieved %d cached messages for session %s", len(messages), session_id)
        return messages

    async def cache_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        cache_key: str,
    ) -> None:
        expires_at = time.monotonic() + self._cache_ttl_seconds
        self._message_cache[cache_key] = (expires_at, messages)
        logger.debug("Cached %d messages for session %s", len(messages), session_id)

    async def create_chat_message(
        self,
        session_id: str,
        message_type: str,
        content: str,
        has_chart: bool = False,
        artifacts: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ChatMessage:
        return ChatMessage(
            session_id=session_id,
            message_type=message_type,
            content=content,
            has_chart=has_chart,
            artifacts=artifacts or {},
            metadata=metadata or {},
            timestamp=timestamp or datetime.now(timezone.utc),
        )

    def invalidate_cache(self, session_id: str) -> None:
        keys_to_remove = [key for key in self._message_cache if session_id in key]
        for key in keys_to_remove:
            self._message_cache.pop(key, None)
        if keys_to_remove:
            logger.debug(
                "Invalidated message cache for session %s: %d entries",
                session_id,
                len(keys_to_remove),
            )

    def build_cache_key(
        self,
        session_id: str,
        limit: int,
        before_timestamp: Optional[datetime] = None,
    ) -> str:
        before_value = before_timestamp.isoformat() if before_timestamp else "none"
        return f"messages_{session_id}_{limit}_{before_value}"

    def get_cache_stats(self) -> dict[str, Any]:
        return {
            "cache_size": len(self._message_cache),
            "ttl_seconds": self._cache_ttl_seconds,
        }


_message_repository: MessageRepository | None = None


def get_message_repository() -> MessageRepository:
    """Return a singleton MessageRepository."""
    global _message_repository
    if _message_repository is None:
        _message_repository = MessageRepository()
    return _message_repository
