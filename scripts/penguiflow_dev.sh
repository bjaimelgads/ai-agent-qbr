#!/usr/bin/env bash
set -euo pipefail

project_root="$(dirname "$0")/.."

if [ -f "$project_root/.env" ]; then
  set -a
  . "$project_root/.env"
  set +a
fi

"$project_root/.venv/bin/penguiflow" dev --project-root "$project_root"
