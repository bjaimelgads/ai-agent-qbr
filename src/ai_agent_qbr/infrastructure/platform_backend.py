"""HTTP gateway for platform backend message persistence."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import httpx

from ai_agent_qbr.infrastructure.databricks import resolve_workspace_token


class PlatformBackendInterface(ABC):
    """Interface for platform backend message operations."""

    @abstractmethod
    async def fetch_messages(
        self,
        session_id: str,
        limit: int = 50,
        before_timestamp: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from platform backend."""

    @abstractmethod
    async def post_message(
        self,
        session_id: str,
        payload: dict[str, Any],
        user_email: Optional[str],
    ) -> None:
        """Persist a message to platform backend."""


def _resolve_platform_url(platform_url: Optional[str]) -> Optional[str]:
    return platform_url or os.getenv("PLATFORM_URL")


def _get_auth_headers() -> dict[str, str]:
    token = resolve_workspace_token(
        host=os.getenv("DATABRICKS_HOST", ""),
        api_base=os.getenv("DATABRICKS_API_BASE", ""),
        token=os.getenv("DATABRICKS_TOKEN"),
        api_key=os.getenv("DATABRICKS_API_KEY"),
        client_id=os.getenv("DATABRICKS_CLIENT_ID"),
        client_secret=os.getenv("DATABRICKS_CLIENT_SECRET"),
    )
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": "ai-agent-qbr/MessageService",
        "Accept": "application/json",
    }


class HttpPlatformBackend(PlatformBackendInterface):
    """HTTP implementation for platform backend operations."""

    def __init__(
        self,
        *,
        platform_url: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._platform_url = _resolve_platform_url(platform_url)
        self._timeout_seconds = timeout_seconds

    async def fetch_messages(
        self,
        session_id: str,
        limit: int = 50,
        before_timestamp: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        if not self._platform_url:
            raise RuntimeError("PLATFORM_URL is not configured")
        url = f"{self._platform_url}/sessions/{session_id}/messages"
        params: dict[str, str] = {}
        if limit:
            params["limit"] = str(limit)
        if before_timestamp:
            params["before_timestamp"] = before_timestamp.isoformat()
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.get(url, params=params or None, headers=_get_auth_headers())
            response.raise_for_status()
            data: Any = response.json() or {}
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            messages = data.get("messages", [])
            return messages if isinstance(messages, list) else []
        return []

    async def post_message(
        self,
        session_id: str,
        payload: dict[str, Any],
        user_email: Optional[str],
    ) -> None:
        if not self._platform_url:
            raise RuntimeError("PLATFORM_URL is not configured")
        url = f"{self._platform_url}/sessions/{session_id}/messages"
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                url,
                params={"user_email": user_email or ""},
                json=payload,
                headers=_get_auth_headers(),
            )
            response.raise_for_status()
