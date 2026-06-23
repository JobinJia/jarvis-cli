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
    raw = ti.get("command")
    cmd = str(raw).strip() if raw else ""
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
    raw = ti.get("pattern")
    pat = str(raw).strip()[:80] if raw else ""
    return f"grep '{pat}'" if pat else "grep"


def _glob(ti: dict[str, Any]) -> str:
    raw = ti.get("pattern")
    pat = str(raw).strip()[:80] if raw else ""
    return f"glob '{pat}'" if pat else "glob"


def _webfetch(ti: dict[str, Any]) -> str:
    raw = ti.get("url")
    url = str(raw).strip()[:120] if raw else ""
    return f"fetch {url}" if url else "fetch"


def _websearch(ti: dict[str, Any]) -> str:
    raw = ti.get("query")
    q = str(raw).strip()[:120] if raw else ""
    return f"search {q!r}" if q else "search"


_AUQ_Q_CAP = 120  # per-question text cap
_AUQ_LABEL_CAP = 60  # per-option label cap


def _askuserquestion(ti: dict[str, Any]) -> str:
    """Summarise an AskUserQuestion tool_input for the phrase router.

    Returns a short structured string ('ask: <q> | options: <l1>; <l2>; ...')
    so the LLM can rephrase question + options into a Jarvis-toned line.
    """
    questions = ti.get("questions")
    if not isinstance(questions, list) or not questions:
        return ""
    first = questions[0] if isinstance(questions[0], dict) else None
    if not first:
        return ""
    q_text = (first.get("question") or "").strip()[:_AUQ_Q_CAP]
    if not q_text:
        return ""
    options = first.get("options") or []
    labels: list[str] = []
    for opt in options[:4]:
        if not isinstance(opt, dict):
            continue
        label = (opt.get("label") or "").strip()[:_AUQ_LABEL_CAP]
        if label:
            labels.append(label)
    parts = [f"ask: {q_text}"]
    if labels:
        parts.append("options: " + "; ".join(labels))
    extra = len(questions) - 1
    if extra > 0:
        parts.append(f"+{extra} more questions")
    return " | ".join(parts)


_EXTRACTORS: dict[str, Callable[[dict[str, Any]], str]] = {
    "Bash": _bash,
    "Write": _write,
    "Edit": _edit,
    "MultiEdit": _edit,
    "Read": _read,
    "Grep": _grep,
    "Glob": _glob,
    "WebFetch": _webfetch,
    "WebSearch": _websearch,
    "AskUserQuestion": _askuserquestion,
}


def extract(tool_name: str | None, tool_input: dict[str, Any] | None) -> str:
    """Return a one-line summary; '' when nothing useful is present."""
    if not tool_input:
        return ""
    if tool_name and tool_name in _EXTRACTORS:
        return _EXTRACTORS[tool_name](tool_input).strip()
    return json.dumps(tool_input, ensure_ascii=False)[:_MAX_RAW]


_FAIL_ERR_CAP = 160  # error-gist cap for the grave failure line


def _error_gist(resp: Any) -> str:
    """Pull the most useful error text out of a PostToolUseFailure
    `tool_response` (CC sends a dict like {"error": "..."} or a bare string)."""
    if isinstance(resp, str):
        return resp.strip()[:_FAIL_ERR_CAP]
    if isinstance(resp, dict):
        for key in ("error", "stderr", "message", "stdout", "output"):
            val = resp.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:_FAIL_ERR_CAP]
    return ""


def extract_failure(
    tool_name: str | None, tool_input: dict[str, Any] | None
) -> str:
    """Summarise a tool_failure event: failed tool + the error gist.

    The hook stashes CC's `tool_response` under tool_input["tool_response"].
    We combine the action (reuse the per-tool extractor on the rest of the
    input, e.g. the bash command) with the error message so the LLM can speak
    a grave, specific line ("the build failed on the test step")."""
    ti = dict(tool_input or {})
    resp = ti.pop("tool_response", None)
    action = extract(tool_name, ti) if ti else ""
    gist = _error_gist(resp)
    parts: list[str] = []
    if tool_name:
        parts.append(f"{tool_name} failed")
    elif not gist and not action:
        return ""
    if action:
        parts.append(action)
    if gist:
        parts.append(gist)
    return ": ".join(parts) if parts else ""
