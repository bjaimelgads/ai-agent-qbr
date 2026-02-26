#!/usr/bin/env python3
"""Generate a fresh Lakebase DB token and persist MIGRATION token env vars."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Lakebase database credential token and update "
            "MIGRATION_TOKEN/MIGRATION_TARGET_TOKEN in an env file."
        )
    )
    parser.add_argument("--env-file", default=".env", help="Env file path (default: .env)")
    parser.add_argument(
        "--instance-name",
        default="",
        help="Optional Lakebase instance name override (e.g. dev-qbr)",
    )
    parser.add_argument(
        "--auth-mode",
        choices=("auto", "pat", "oauth"),
        default="auto",
        help=(
            "Auth mode for Databricks SDK. "
            "'auto' prefers OAuth client credentials when configured, otherwise PAT."
        ),
    )
    return parser.parse_args()


def _update_env_tokens(env_file: Path, token: str) -> None:
    if not env_file.exists():
        raise SystemExit(f"Env file not found: {env_file}")

    lines = env_file.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen_migration_token = False
    seen_target_token = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("MIGRATION_TOKEN="):
            if not seen_migration_token:
                out.append(f"MIGRATION_TOKEN={token}")
                seen_migration_token = True
            continue
        if stripped.startswith("MIGRATION_TARGET_TOKEN="):
            if not seen_target_token:
                out.append("MIGRATION_TARGET_TOKEN=${MIGRATION_TOKEN}")
                seen_target_token = True
            continue
        out.append(line)

    if not seen_migration_token:
        out.append(f"MIGRATION_TOKEN={token}")
    if not seen_target_token:
        out.append("MIGRATION_TARGET_TOKEN=${MIGRATION_TOKEN}")

    env_file.write_text("\n".join(out) + "\n", encoding="utf-8")


def _normalize_host(value: str) -> str:
    host = value.strip()
    if host and not host.startswith(("https://", "http://")):
        host = f"https://{host}"
    return host


def _strip_dns_to_uid(migration_target_host: str) -> str:
    host = migration_target_host.strip()
    if not host:
        return ""
    host = host.removeprefix("https://").removeprefix("http://")
    if ".database.cloud.databricks.com" not in host:
        return ""
    return host.split(".database.cloud.databricks.com")[0].removeprefix("instance-")


def _resolve_instance_name(
    *,
    client: WorkspaceClient,
    configured_name: str,
    migration_target_host: str,
) -> str:
    if configured_name and not configured_name.startswith("instance-"):
        return configured_name

    uid_from_host = _strip_dns_to_uid(migration_target_host)

    instances = list(client.database.list_database_instances())
    if configured_name:
        for inst in instances:
            if getattr(inst, "name", "") == configured_name:
                return configured_name

    if uid_from_host:
        for inst in instances:
            if getattr(inst, "uid", "") == uid_from_host:
                return getattr(inst, "name", "") or ""

    raise SystemExit(
        "Could not resolve Lakebase instance name. Set QBR_LAKEBASE_DB_INSTANCE to the "
        "instance name (e.g. dev-qbr) or pass --instance-name."
    )


def main() -> int:
    args = _parse_args()
    env_file = Path(args.env_file).resolve()
    load_dotenv(env_file, override=True)

    host = _normalize_host(
        os.getenv("QBR_LAKEBASE_DATABRICKS_HOST", "")
        or os.getenv("DATABRICKS_HOST", "")
    )
    pat_token = (
        os.getenv("DATABRICKS_API_KEY", "")
        or os.getenv("DATABRICKS_TOKEN", "")
        or os.getenv("QBR_LAKEBASE_TOKEN", "")
    ).strip()
    client_id = (os.getenv("DATABRICKS_CLIENT_ID", "") or "").strip()
    client_secret = (os.getenv("DATABRICKS_CLIENT_SECRET", "") or "").strip()
    if not host:
        raise SystemExit(
            "Missing Databricks host. Set DATABRICKS_HOST or QBR_LAKEBASE_DATABRICKS_HOST."
        )

    if args.auth_mode == "pat":
        if not pat_token:
            raise SystemExit(
                "Missing PAT token. Set DATABRICKS_API_KEY (or DATABRICKS_TOKEN/QBR_LAKEBASE_TOKEN)."
            )
        client = WorkspaceClient(host=host, token=pat_token, auth_type="pat")
    elif args.auth_mode == "oauth":
        if not client_id or not client_secret:
            raise SystemExit(
                "Missing OAuth credentials. Set DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET."
            )
        client = WorkspaceClient(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
        )
    else:
        # Auto mode: prefer OAuth client credentials if available, fallback to PAT.
        if client_id and client_secret:
            client = WorkspaceClient(
                host=host,
                client_id=client_id,
                client_secret=client_secret,
            )
        elif pat_token:
            client = WorkspaceClient(host=host, token=pat_token, auth_type="pat")
        else:
            raise SystemExit(
                "Missing Databricks auth. Provide either DATABRICKS_CLIENT_ID + "
                "DATABRICKS_CLIENT_SECRET, or DATABRICKS_API_KEY "
                "(or DATABRICKS_TOKEN/QBR_LAKEBASE_TOKEN)."
            )
    configured_instance = (args.instance_name or os.getenv("QBR_LAKEBASE_DB_INSTANCE", "")).strip()
    migration_target_host = os.getenv("MIGRATION_TARGET_HOST", "")
    instance_name = _resolve_instance_name(
        client=client,
        configured_name=configured_instance,
        migration_target_host=migration_target_host,
    )

    credential = client.database.generate_database_credential(
        request_id=str(uuid.uuid4()),
        instance_names=[instance_name],
    )
    db_token = (getattr(credential, "token", None) or "").strip()
    if not db_token:
        raise SystemExit("Credential API returned empty token.")

    _update_env_tokens(env_file, db_token)
    print(f"Updated MIGRATION_TOKEN in {env_file} for instance '{instance_name}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
