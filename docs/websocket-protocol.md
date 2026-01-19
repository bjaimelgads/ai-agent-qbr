# WebSocket Protocol

Endpoint: `/ws/chat/{session_id}`

## Client -> Server

```json
{ "message": "Hello", "metadata": { "document_id": "doc-1" } }
```

Optional keepalive:

```json
{ "type": "ping" }
```

## Server -> Client DTOs

Ready:

```json
{ "status": "ready", "session_id": "abc" }
```

Thinking:

```json
{ "status": "thinking", "message": "Planning response", "step": 1 }
```

User message echo:

```json
{ "status": "user_message", "data": { "content": "Hello", "session_id": "abc" } }
```

Final:

```json
{ "status": "final", "data": { "content": "..." }, "citations": [] }
```

Error:

```json
{ "status": "error", "message": "..." }
```

Keepalive:

```json
{ "type": "ping" }
{ "type": "pong" }
```

## Notes

- The server sends periodic `ping` messages. Clients should reply with `pong`.
- Planner events are mapped into `thinking` updates; final output is always `final`.
