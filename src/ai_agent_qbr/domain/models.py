"""Domain models for session handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SessionContext:
    tenant_id: str
    user_id: str
    session_id: str


@dataclass(frozen=True)
class MemoryInteraction:
    user_prompt: str
    agent_response: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    session_id: str
    message_type: str
    content: str
    timestamp: datetime
    has_chart: bool = False
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
