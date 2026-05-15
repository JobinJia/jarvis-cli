"""Per-tool extractor: turns raw tool_input into a short summary string.

The summary is what the LLM-side prompt sees instead of the full
tool_input dict. Keeps the prompt small, predictable, and free of
nuisance fields (Write/Edit content blobs, etc.).
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

_MAX_RAW = 200


def _basename(path: str) -> str:
    return os.path.basename(path.rstrip("/")) or path


def _bash(ti: dict[str, Any]) -> str:
    cmd = str(ti.get("command", "")).strip()
    return cmd[:_MAX_RAW]


def _write(ti: dict[str, Any]) -> str:
    fp = str(ti.get("file_path", ""))
    return f"write {_basename(fp)}" if fp else "write"


def _edit(ti: dict[str, Any]) -> str:
    fp = str(ti.get("file_path", ""))
    return f"edit {_basename(fp)}" if fp else "edit"


def _read(ti: dict[str, Any]) -> str:
    fp = str(ti.get("file_path", ""))
    return f"read {_basename(fp)}" if fp else "read"


def _grep(ti: dict[str, Any]) -> str:
    pat = str(ti.get("pattern", ""))[:80]
    return f"grep '{pat}'" if pat else "grep"


def _glob(ti: dict[str, Any]) -> str:
    pat = str(ti.get("pattern", ""))[:80]
    return f"glob '{pat}'" if pat else "glob"


def _webfetch(ti: dict[str, Any]) -> str:
    url = str(ti.get("url", ""))[:120]
    return f"fetch {url}" if url else "fetch"


_EXTRACTORS: dict[str, Callable[[dict[str, Any]], str]] = {
    "Bash": _bash,
    "Write": _write,
    "Edit": _edit,
    "MultiEdit": _edit,
    "Read": _read,
    "Grep": _grep,
    "Glob": _glob,
    "WebFetch": _webfetch,
    "WebSearch": _webfetch,
}


def extract(tool_name: str | None, tool_input: dict[str, Any] | None) -> str:
    """Return a one-line summary; '' when nothing useful is present."""
    if not tool_input:
        return ""
    if tool_name and tool_name in _EXTRACTORS:
        return _EXTRACTORS[tool_name](tool_input).strip()
    return json.dumps(tool_input, ensure_ascii=False)[:_MAX_RAW]
