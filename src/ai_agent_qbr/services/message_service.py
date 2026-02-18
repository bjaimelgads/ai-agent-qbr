"""Message service for platform backend persistence and retrieval."""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

from ai_agent_qbr.domain.models import ChatMessage
from ai_agent_qbr.infrastructure.message_repository import (
    MessageRepositoryInterface,
    get_message_repository,
)
from ai_agent_qbr.infrastructure.platform_backend import (
    HttpPlatformBackend,
    PlatformBackendInterface,
)

logger = logging.getLogger("ai_agent_qbr.platform")


class MessageService:
    """Service for message persistence and caching."""

    def __init__(
        self,
        repository: Optional[MessageRepositoryInterface] = None,
        backend: Optional[PlatformBackendInterface] = None,
    ) -> None:
        self._repository = repository or get_message_repository()
        self._backend = backend or HttpPlatformBackend()
        logger.info("MessageService initialized")

    async def get_messages_from_cache(
        self,
        session_id: str,
        limit: int = 50,
        before_timestamp: Optional[datetime] = None,
    ) -> Optional[list[dict[str, Any]]]:
        messages = await self._repository.get_messages(
            session_id=session_id,
            limit=limit,
            before_timestamp=before_timestamp,
        )
        return messages if messages else None

    async def get_messages_from_backend(
        self,
        session_id: str,
        limit: int = 50,
        before_timestamp: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        messages = await self._backend.fetch_messages(
            session_id=session_id,
            limit=limit,
            before_timestamp=before_timestamp,
        )
        cache_key = self._repository.build_cache_key(session_id, limit, before_timestamp)
        await self._repository.cache_messages(session_id, messages, cache_key)
        return messages

    async def persist_message(
        self,
        *,
        session_id: str,
        message_type: str,
        content: str,
        user_email: Optional[str] = None,
        artifacts: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> ChatMessage:
        self._validate_message(session_id, message_type, content)
        payload = self._prepare_payload(
            session_id=session_id,
            message_type=message_type,
            content=content,
            artifacts=artifacts,
            metadata=metadata,
            timestamp=timestamp,
        )

        self._repository.invalidate_cache(session_id)
        asyncio.create_task(self._post_message_to_backend(session_id, payload, user_email))

        return await self._repository.create_chat_message(
            session_id=session_id,
            message_type=message_type,
            content=content,
            has_chart=bool(artifacts and artifacts.get("type") == "echarts"),
            artifacts=payload["artifacts"],
            metadata=payload["metadata"],
            timestamp=timestamp or datetime.now(timezone.utc),
        )

    async def _post_message_to_backend(
        self,
        session_id: str,
        payload: dict[str, Any],
        user_email: Optional[str],
    ) -> None:
        message_type = payload.get("message_type")
        trace_id = None
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            trace_id = metadata.get("trace_id")
        logger.info(
            "platform_message status=send_attempt session_id=%s message_type=%s trace_id=%s",
            session_id,
            message_type,
            trace_id,
        )
        try:
            await self._backend.post_message(
                session_id=session_id,
                payload=self._jsonify_payload(payload),
                user_email=user_email,
            )
            logger.info(
                "platform_message status=success session_id=%s message_type=%s trace_id=%s",
                session_id,
                message_type,
                trace_id,
            )
        except Exception as exc:
            logger.warning(
                "platform_message status=failure session_id=%s message_type=%s trace_id=%s error=%s",
                session_id,
                message_type,
                trace_id,
                exc,
            )

    def _prepare_payload(
        self,
        *,
        session_id: str,
        message_type: str,
        content: str,
        artifacts: Optional[dict[str, Any]],
        metadata: Optional[dict[str, Any]],
        timestamp: Optional[datetime],
    ) -> dict[str, Any]:
        has_chart = bool(artifacts and artifacts.get("type") == "echarts")
        payload = {
            "session_id": session_id,
            "message_type": message_type,
            "content": content,
            "has_chart": has_chart,
            "artifacts": artifacts or {},
            "metadata": metadata or {},
        }
        if timestamp:
            payload["timestamp"] = timestamp
        return payload

    def _validate_message(self, session_id: str, message_type: str, content: str) -> None:
        if not session_id or not session_id.strip():
            raise ValueError("Session ID is required")
        if message_type not in {"user", "assistant"}:
            raise ValueError("message_type must be 'user' or 'assistant'")
        if not content or not content.strip():
            raise ValueError("Message content is required")

    def _jsonify_payload(self, value: Any) -> Any:
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {key: self._jsonify_payload(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._jsonify_payload(item) for item in value]
        if isinstance(value, tuple):
            return [self._jsonify_payload(item) for item in value]
        return value

    async def get_cache_stats(self) -> dict[str, Any]:
        return self._repository.get_cache_stats()


_message_service: MessageService | None = None


def get_message_service() -> MessageService:
    """Return a singleton MessageService."""
    global _message_service
    if _message_service is None:
        _message_service = MessageService()
    return _message_service
