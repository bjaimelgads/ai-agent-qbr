"""Databricks workspace helpers for guardrail auth."""

from __future__ import annotations

import os

from databricks.sdk import WorkspaceClient


def _is_running_locally() -> bool:
    return os.getenv("LOCAL_DEVELOPMENT", "false").lower() == "true"


def _resolve_workspace_host(*, host: str, api_base: str) -> str:
    if host:
        return host
    if "/serving-endpoints" in api_base:
        return api_base.split("/serving-endpoints", 1)[0]
    return api_base


def _extract_workspace_token(workspace_client: WorkspaceClient) -> str:
    try:
        if workspace_client.config.auth_type == "oauth-m2m":
            auth_header = workspace_client.config.authenticate() or {}
            bearer = auth_header.get("Authorization", "")
            if bearer.startswith("Bearer "):
                return bearer.replace("Bearer ", "", 1)
        token = workspace_client.config.token or ""
        if token:
            return token
        auth_header = workspace_client.config.authenticate() or {}
        bearer = auth_header.get("Authorization", "")
        if bearer.startswith("Bearer "):
            return bearer.replace("Bearer ", "", 1)
    except Exception:
        return ""
    return ""


def resolve_workspace_token(
    *,
    host: str,
    api_base: str,
    token: str | None = None,
    api_key: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> str:
    """Resolve a workspace auth token for Databricks API calls."""
    if _is_running_locally():
        resolved_host = _resolve_workspace_host(host=host, api_base=api_base)
        if client_id and client_secret:
            workspace_client = WorkspaceClient(
                host=resolved_host,
                client_id=client_id,
                client_secret=client_secret,
            )
            return _extract_workspace_token(workspace_client)
        return token or api_key or ""

    workspace_client = WorkspaceClient()
    return _extract_workspace_token(workspace_client)


__all__ = ["resolve_workspace_token"]
