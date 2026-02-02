#!/usr/bin/env python3
"""CLI for interacting with ai-agent-qbr over WebSockets.

Supports legacy (OUTPUT_PROTOCOL=websocket) and AG-UI (OUTPUT_PROTOCOL=agui)
message formats over the same /ws/chat/{session_id} endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import sys
from typing import Any

import websockets


def _build_ws_url(base_url: str, session_id: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/ws/chat/{session_id}"


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=True))


def _extract_text_from_agui(event: dict[str, Any]) -> str | None:
    event_type = str(event.get("type") or "").upper()
    if event_type == "TEXT_MESSAGE_CONTENT":
        return str(event.get("delta") or "")
    if event_type == "TEXT_MESSAGE_END":
        return "\n"
    if event_type == "RUN_ERROR":
        return f"\n[error] {event.get('message', '')}\n"
    return None


async def _legacy_roundtrip(
    ws: websockets.WebSocketClientProtocol,
    message: str,
    metadata: dict[str, Any] | None,
    raw: bool,
    timeout_seconds: float | None,
) -> int:
    await ws.send(json.dumps({"message": message, "metadata": metadata or {}}))
    printed_prefix = False
    while True:
        try:
            payload = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_seconds))
        except asyncio.TimeoutError:
            print("[error] timed out waiting for server response", file=sys.stderr)
            return 1
        if isinstance(payload, dict) and payload.get("type") == "ping":
            await ws.send(json.dumps({"type": "pong"}))
            continue
        if raw:
            _print_json(payload)
            continue
        status = payload.get("status") if isinstance(payload, dict) else None
        if status in {"ready", "user_message"}:
            continue
        if status == "thinking":
            msg = payload.get("message")
            if msg:
                print(f"[thinking] {msg}")
        elif status == "partial":
            data = payload.get("data") or {}
            text = data.get("content") or ""
            if text:
                if not printed_prefix:
                    sys.stdout.write("assistant> ")
                    printed_prefix = True
                sys.stdout.write(text)
                sys.stdout.flush()
        elif status == "final":
            data = payload.get("data") or {}
            text = data.get("content") or ""
            if text:
                if not printed_prefix:
                    sys.stdout.write("assistant> ")
                print(text)
            else:
                print("assistant> ")
            return 0
        elif status == "error":
            msg = payload.get("message")
            if msg:
                print(f"[error] {msg}", file=sys.stderr)
            return 1
        else:
            if payload:
                _print_json(payload)


async def _agui_roundtrip(
    ws: websockets.WebSocketClientProtocol,
    message: str,
    metadata: dict[str, Any] | None,
    raw: bool,
    timeout_seconds: float | None,
) -> int:
    payload = {"message": message}
    if metadata:
        payload["metadata"] = metadata

    await ws.send(json.dumps(payload))
    printed_prefix = False
    while True:
        try:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_seconds))
        except asyncio.TimeoutError:
            print("[error] timed out waiting for server response", file=sys.stderr)
            return 1
        if isinstance(event, dict) and event.get("type") == "ping":
            await ws.send(json.dumps({"type": "pong"}))
            continue

        event_type = str(event.get("type") or "").upper() if isinstance(event, dict) else ""
        if raw:
            _print_json(event)
        else:
            text = _extract_text_from_agui(event if isinstance(event, dict) else {})
            if text:
                if not printed_prefix:
                    sys.stdout.write("assistant> ")
                    printed_prefix = True
                sys.stdout.write(text)
                sys.stdout.flush()

        if event_type in {"RUN_FINISHED", "RUN_ERROR", "RUN_CANCELLED"}:
            if event_type == "RUN_ERROR":
                return 1
            if not raw and printed_prefix:
                sys.stdout.write("\n")
            return 0


async def _run(args: argparse.Namespace) -> int:
    session_id = args.session_id or secrets.token_hex(8)
    url = _build_ws_url(args.base_url, session_id)

    metadata = {}
    if args.tenant_id:
        metadata["tenant_id"] = args.tenant_id
    if args.user_id:
        metadata["user_id"] = args.user_id
    if args.document_id:
        metadata["document_id"] = args.document_id

    async def send_once(ws: websockets.WebSocketClientProtocol, text: str) -> int:
        if args.protocol == "legacy":
            return await _legacy_roundtrip(ws, text, metadata or None, args.raw, args.timeout)
        return await _agui_roundtrip(ws, text, metadata or None, args.raw, args.timeout)

    if args.repl or not args.message:
        async with websockets.connect(url, ping_interval=None) as ws:
            print(f"Connected to {url}")
            print("Type 'exit' or 'quit' to stop.")
            while True:
                user_text = await asyncio.to_thread(lambda: input("user> ").strip())
                if not user_text:
                    continue
                if user_text.lower() in {"exit", "quit"}:
                    return 0
                code = await send_once(ws, user_text)
                if code != 0:
                    return code
            return 0

    message = args.message
    if not message:
        message = sys.stdin.read().strip()
    if not message:
        print("Missing message (use --message or stdin)", file=sys.stderr)
        return 2
    async with websockets.connect(url, ping_interval=None) as ws:
        return await send_once(ws, message)


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket CLI for ai-agent-qbr")
    parser.add_argument(
        "--base-url",
        default="ws://localhost:8000",
        help="Base URL for the server (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID to use (default: random)",
    )
    parser.add_argument(
        "--protocol",
        choices=["legacy", "agui"],
        default="legacy",
        help="Protocol to use over WebSocket (default: legacy)",
    )
    parser.add_argument("--message", help="Message to send (or provide via stdin)")
    parser.add_argument("--repl", action="store_true", help="Interactive chat mode")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for each server response (default: 30)",
    )
    parser.add_argument("--tenant-id", help="Tenant ID metadata")
    parser.add_argument("--user-id", help="User ID metadata")
    parser.add_argument("--document-id", help="Document ID metadata")
    parser.add_argument("--raw", action="store_true", help="Print raw JSON payloads")

    args = parser.parse_args()
    try:
        exit_code = asyncio.run(_run(args))
    except KeyboardInterrupt:
        exit_code = 130
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
