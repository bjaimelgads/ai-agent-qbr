#!/usr/bin/env python3
"""Update only MIGRATION token vars in .env using a user-provided token."""

from __future__ import annotations

import argparse
import os
import getpass
from pathlib import Path

from dotenv import load_dotenv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set MIGRATION_TOKEN in an env file from a provided value."
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to env file to update (default: .env)",
    )
    parser.add_argument(
        "--token",
        help="Migration token value to write. If omitted, MIGRATION_TOKEN from env is used.",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Prompt securely for the token (input hidden).",
    )
    parser.add_argument(
        "--print-token",
        action="store_true",
        help="Print token to stdout (avoid in shared terminals/logs).",
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


def main() -> int:
    args = _parse_args()
    env_file = Path(args.env_file).resolve()

    load_dotenv(env_file, override=True)
    token = (args.token or "").strip()
    if args.prompt and not token:
        token = getpass.getpass("Enter MIGRATION_TOKEN: ").strip()
    if not token:
        token = (os.getenv("MIGRATION_TOKEN") or "").strip()
    if not token:
        raise SystemExit(
            "Missing token. Provide --token or set MIGRATION_TOKEN in the environment."
        )
    _update_env_tokens(env_file, token)

    print(f"Updated token vars in: {env_file}")
    if args.print_token:
        print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
