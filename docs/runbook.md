# Runbook

## Common Failures

### No retrieval results
- Confirm `DATABASE_URL` points to the correct `qbr_intelligence.db`.
- Verify chunk embeddings exist (`chunks.embedding` not null).
- Ensure `EMBEDDINGS_BACKEND` matches the embedding model stored in the DB.
- Temporarily set `EMBEDDINGS_BACKEND=hash` for deterministic testing.
- If using FAISS, ensure `FAISS_DIR` contains `index.faiss` and `index_ids.json`.
- Enable `FAISS_AUTO_BUILD=true` to build at startup when missing.

### Embedding mismatch
- Check the stored `chunks.embedding_model` values.
- Set `EMBEDDINGS_MODEL` to match the extraction model.
- Re-run extraction with consistent embedding settings.
- Set `FAISS_EMBEDDING_MODEL_FILTER` when building the FAISS index.

### WebSocket connection issues
- Ensure the server is running and `PORT` is open.
- Verify the endpoint: `/ws/chat/{session_id}`.
- Check for proxy/websocket timeouts.

### AG-UI payload shape mismatch
- Confirm `OUTPUT_PROTOCOL=agui`.
- Verify client supports AG-UI event types (`RUN_STARTED`, `TEXT_MESSAGE_CONTENT`, etc.).
- For legacy consumers, set `OUTPUT_PROTOCOL=websocket`.

## Debugging Tips
- Enable logs: `LOG_LEVEL=debug`.
- Use the CLI test harness: `python -m ai_agent_qbr`.
- Run `pytest tests/test_websocket_agui_e2e.py` to validate streaming.
