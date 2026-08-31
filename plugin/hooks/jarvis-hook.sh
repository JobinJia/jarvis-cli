#!/usr/bin/env bash
# Thin plugin wrapper: forward stdin to jarvis-hook if installed.
# UserPromptSubmit relies on stdout for skill injection — exec preserves
# the stdout pipe so additionalContext flows back to Claude Code.
if command -v jarvis-hook &>/dev/null; then
    exec jarvis-hook "$@"
elif command -v uvx &>/dev/null; then
    exec uvx --from jarvis jarvis-hook "$@"
fi
# Silently exit if not installed — never block Claude Code.
exit 0
