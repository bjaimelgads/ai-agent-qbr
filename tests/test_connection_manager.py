import asyncio

from ai_agent_qbr.transport.websocket.connection_manager import ConnectionManager


class FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload):
        self.sent.append(payload)


def test_connection_manager_connect_and_send():
    manager = ConnectionManager()
    socket = FakeWebSocket()

    asyncio.run(manager.connect("session", socket))
    assert socket.accepted is True

    asyncio.run(manager.send_json("session", {"type": "ready"}))
    assert socket.sent == [{"type": "ready"}]
