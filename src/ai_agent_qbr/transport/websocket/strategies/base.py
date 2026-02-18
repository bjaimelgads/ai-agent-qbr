"""WebSocket output strategies for different frontend protocols."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class WebsocketOutputStrategy(Protocol):
    """Protocol for handling WebSocket message flows."""

    async def on_connect(self, session_id: str) -> None:
        ...

    async def on_message(self, session_id: str, message: Any) -> None:
        ...

    async def on_disconnect(self, session_id: str) -> None:
        ...


SendJson = Callable[[str, Any], Awaitable[None]]
