"""Databricks app entrypoint for ai-agent-qbr."""

import os
import sys

if not (os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DATABRICKS_APP_NAME")):
    from dotenv import load_dotenv

    load_dotenv()

ROOT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import uvicorn


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    uvicorn.run(
        "ai_agent_qbr.api.app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
