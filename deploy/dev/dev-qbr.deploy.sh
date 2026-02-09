#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPLOY_DIR="$ROOT_DIR/deployment_files"

# Set up deployment folder
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

# Copy common files for all deployments
uv export --format requirements-txt --no-hashes --no-dev --extra retrieval --output-file "$DEPLOY_DIR/requirements.txt"
cp "$ROOT_DIR/pyproject.toml" "$DEPLOY_DIR/"
if [ -f "$ROOT_DIR/uv.lock" ]; then
  cp "$ROOT_DIR/uv.lock" "$DEPLOY_DIR/"
fi
cp -rf "$ROOT_DIR/src" "$DEPLOY_DIR/src"
cp "$ROOT_DIR/app.py" "$DEPLOY_DIR"

# Copy prebuilt FAISS index (assumed to be in $ROOT_DIR/data/faiss)

# Ship local SQLite DB + FAISS index if requested
DEPLOY_DB_BUNDLE="${DEPLOY_DB_BUNDLE:-false}"
DEPLOY_FAISS_BUNDLE="${DEPLOY_FAISS_BUNDLE:-false}"

if [ "$DEPLOY_DB_BUNDLE" = "true" ]; then
  if [ -f "$ROOT_DIR/qbr_intelligence.db" ]; then
    cp "$ROOT_DIR/qbr_intelligence.db" "$DEPLOY_DIR/qbr_intelligence.db"
  else
    echo "WARN: Missing $ROOT_DIR/qbr_intelligence.db; skipping DB bundle." >&2
  fi
else
  echo "Skipping DB bundle (DEPLOY_DB_BUNDLE=false)."
fi

if [ "$DEPLOY_FAISS_BUNDLE" = "true" ]; then
  if [ -d "$ROOT_DIR/data/faiss" ]; then
    mkdir -p "$DEPLOY_DIR/data"
    cp -rf "$ROOT_DIR/data/faiss" "$DEPLOY_DIR/data/"
    if [ ! -f "$DEPLOY_DIR/data/faiss/index.faiss" ] || [ ! -f "$DEPLOY_DIR/data/faiss/index_ids.json" ]; then
      echo "ERROR: FAISS index files missing in deployment bundle." >&2
      exit 1
    fi
  else
    echo "ERROR: Missing $ROOT_DIR/data/faiss; cannot deploy FAISS index." >&2
    exit 1
  fi
else
  echo "Skipping FAISS bundle (DEPLOY_FAISS_BUNDLE=false)."
fi

# Upload files to databricks
cd "$DEPLOY_DIR"

# Sync
cp "$ROOT_DIR/deploy/dev/dev-qbr.app.yaml" app.yaml
echo "Deployment bundle contents:"
ls -la "$DEPLOY_DIR"
if [ -d "$DEPLOY_DIR/data" ]; then
  ls -la "$DEPLOY_DIR/data"
fi
if [ -d "$DEPLOY_DIR/data/faiss" ]; then
  ls -la "$DEPLOY_DIR/data/faiss"
fi

databricks sync . /Workspace/Users/damian.beltritti@consultants.lgads.tv/databricks_apps/dev-qbr-agent

# Back to root
cd "$ROOT_DIR"

# Deploy dev version
databricks apps deploy dev-qbr-agent --source-code-path /Workspace/Users/damian.beltritti@consultants.lgads.tv/databricks_apps/dev-qbr-agent

# Cleanup deployment files
rm -rf "$DEPLOY_DIR"
