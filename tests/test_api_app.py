from ai_agent_qbr.api.app import app


def test_websocket_route_registered():
    paths = [route.path for route in app.router.routes]
    assert "/ws/chat/{session_id}" in paths
