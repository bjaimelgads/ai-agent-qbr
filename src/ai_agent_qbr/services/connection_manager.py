"""Connection manager modeled after ai-agent-reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket


@dataclass
class ConnectionManager:
    active_connections: dict[str, WebSocket] = field(default_factory=dict)

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        self.active_connections.pop(session_id, None)

    async def send_json(self, session_id: str, payload: Any) -> None:
        websocket = self.active_connections.get(session_id)
        if websocket is None:
            return
        await websocket.send_json(payload)
