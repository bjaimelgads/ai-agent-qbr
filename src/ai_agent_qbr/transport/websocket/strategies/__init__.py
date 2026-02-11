"""WebSocket output strategy selection."""

from __future__ import annotations

from typing import Any

from ai_agent_qbr.config import Config
from ai_agent_qbr.transport.websocket.connection_manager import ConnectionManager

from .agui import AguiWebsocketOutputStrategy
from .base import WebsocketOutputStrategy
from .legacy import LegacyWebsocketOutputStrategy


def build_websocket_strategy(
    *,
    config: Config,
    manager: ConnectionManager,
    send_message_use_case: Any,
    session_registry: Any,
    logger: Any,
) -> WebsocketOutputStrategy:
    sender = manager.send_json
    if config.output_protocol == "agui":
        return AguiWebsocketOutputStrategy(
            sender=sender,
            send_message=send_message_use_case,
            session_registry=session_registry,
            logger=logger,
            reasoning_source=config.agui_reasoning_source,
        )
    return LegacyWebsocketOutputStrategy(
        sender=sender,
        send_message=send_message_use_case,
        session_registry=session_registry,
        logger=logger,
    )


__all__ = ["WebsocketOutputStrategy", "build_websocket_strategy"]
