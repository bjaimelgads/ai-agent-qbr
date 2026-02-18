import asyncio
from datetime import datetime, timezone

import pytest

from ai_agent_qbr.infrastructure.message_repository import MessageRepository
from ai_agent_qbr.infrastructure.platform_backend import PlatformBackendInterface
from ai_agent_qbr.services.message_service import MessageService


class DummyBackend(PlatformBackendInterface):
    def __init__(self) -> None:
        self.posts: list[dict[str, object]] = []
        self.post_event = asyncio.Event()
        self.fetch_responses: list[list[dict[str, object]]] = []

    async def fetch_messages(self, session_id: str, limit: int = 50, before_timestamp=None):
        if self.fetch_responses:
            return self.fetch_responses.pop(0)
        return []

    async def post_message(self, session_id: str, payload: dict[str, object], user_email: str | None):
        self.posts.append(
            {
                "session_id": session_id,
                "payload": payload,
                "user_email": user_email,
            }
        )
        self.post_event.set()


@pytest.mark.asyncio
async def test_persist_message_posts_in_background():
    repo = MessageRepository(cache_ttl_seconds=1)
    backend = DummyBackend()
    service = MessageService(repository=repo, backend=backend)
    timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)

    message = await service.persist_message(
        session_id="session-1",
        message_type="user",
        content="Hello",
        user_email="user@example.com",
        metadata={"trace_id": "trace-1"},
        timestamp=timestamp,
    )

    await asyncio.wait_for(backend.post_event.wait(), timeout=1.0)

    assert message.session_id == "session-1"
    assert message.message_type == "user"
    assert message.content == "Hello"
    assert message.metadata["trace_id"] == "trace-1"

    assert backend.posts
    post = backend.posts[0]
    assert post["session_id"] == "session-1"
    payload = post["payload"]
    assert payload["message_type"] == "user"
    assert payload["content"] == "Hello"
    assert payload["metadata"]["trace_id"] == "trace-1"


@pytest.mark.asyncio
async def test_get_messages_from_backend_caches():
    repo = MessageRepository(cache_ttl_seconds=60)
    backend = DummyBackend()
    backend.fetch_responses.append([{"message_type": "assistant", "content": "Hi"}])
    service = MessageService(repository=repo, backend=backend)

    messages = await service.get_messages_from_backend("session-2", limit=10)
    assert messages == [{"message_type": "assistant", "content": "Hi"}]

    cached = await service.get_messages_from_cache("session-2", limit=10)
    assert cached == messages
