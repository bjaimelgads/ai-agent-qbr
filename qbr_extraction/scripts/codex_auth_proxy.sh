#!/usr/bin/env bash
set -euo pipefail

# Forward host callbacks to Codex CLI bound on container localhost.
# Usage: run in the container before starting `codex login`.

socat TCP-LISTEN:1456,fork,bind=0.0.0.0 TCP:127.0.0.1:1455
