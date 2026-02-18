import pytest

from ai_agent_qbr.config import Config
from ai_agent_qbr.transport.websocket.service import WebsocketChatService


class FakeManager:
    def __init__(self):
        self.active_connections = {"session": object()}
        self.sent = []

    async def send_json(self, session_id, payload):
        self.sent.append((session_id, payload))

    async def connect(self, session_id, websocket):
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id):
        self.active_connections.pop(session_id, None)


class FakeUseCase:
    async def execute(self, *args, **kwargs):
        return None


class FakeOutputStrategy:
    def __init__(self):
        self.messages = []

    async def on_connect(self, session_id):
        return None

    async def on_message(self, session_id, message):
        self.messages.append((session_id, message))

    async def on_disconnect(self, session_id):
        return None


@pytest.mark.asyncio
async def test_websocket_service_dispatches_message():
    service = WebsocketChatService(
        connection_manager=FakeManager(),
        start_session=FakeUseCase(),
        close_session=FakeUseCase(),
        config=Config(),
        output_strategy=FakeOutputStrategy(),
        telemetry_factory=lambda: None,
    )

    await service._handle_message("session", {"message": "hello"})
    assert service._output_strategy.messages == [("session", {"message": "hello"})]
