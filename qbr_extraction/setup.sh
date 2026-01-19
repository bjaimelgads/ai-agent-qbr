#!/usr/bin/env bash
set -euo pipefail

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
source ~/.bashrc

# Force a Python version compatible with onnxruntime wheels
uv python install 3.12
UV_VENV_CLEAR=1 uv venv --python 3.12
export UV_PYTHON=3.12
uv sync

# Install ruff
uv tool install ruff

# Install nvm and Node.js
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash
source "$HOME/.nvm/nvm.sh"
nvm install 22

# Install Claude
npm install -g @anthropic-ai/claude-code

# Install Codex CLI (OpenAI)
npm install -g @openai/codex

# Networking helper for OAuth callbacks
sudo apt-get update
sudo apt-get install -y socat

# Install Databricks CLI
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sudo sh

# Configure Databricks CLI to use local repo config if present
echo "export DATABRICKS_CONFIG_FILE=$(pwd)/.databrickscfg" >> ~/.bashrc

echo "Setup complete. Open a new shell or 'source ~/.bashrc' to apply environment updates."
