#!/usr/bin/env bash
# Thin plugin wrapper: forward stdin to jarvis-cli-hook if installed.
# UserPromptSubmit relies on stdout for skill injection — exec preserves
# the stdout pipe so additionalContext flows back to Claude Code.
if command -v jarvis-cli-hook &>/dev/null; then
    exec jarvis-cli-hook "$@"
elif command -v uvx &>/dev/null; then
    exec uvx --from jarvis-cli jarvis-cli-hook "$@"
fi
# Silently exit if not installed — never block Claude Code.
exit 0
