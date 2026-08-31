#!/usr/bin/env bash
set -euo pipefail
echo "Installing jarvis..."
if command -v uv &>/dev/null; then
    uv tool install jarvis
elif command -v pipx &>/dev/null; then
    pipx install jarvis
else
    pip install --user jarvis
fi
jarvis install --non-interactive
echo "Jarvis is ready, sir."
