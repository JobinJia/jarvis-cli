#!/usr/bin/env bash
set -euo pipefail
echo "Installing jarvis-cli..."
if command -v uv &>/dev/null; then
    uv tool install jarvis-cli
elif command -v pipx &>/dev/null; then
    pipx install jarvis-cli
else
    pip install --user jarvis-cli
fi
jarvis-cli install --non-interactive
echo "Jarvis is ready, sir."
