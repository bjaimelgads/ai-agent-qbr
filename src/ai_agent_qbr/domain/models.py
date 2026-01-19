"""Domain models for session handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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
