"""Thin client invoked by Claude Code Notification hook.

Reads JSON payload from stdin, writes a single NDJSON line over the
configured Unix socket, and exits. Must never raise to stdout — Claude
Code reads stdout for hook decisions.
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import IO

from .config import DEFAULT_CONFIG_PATH, load_config

# CC's PreToolUse(AskUserQuestion) payload has no `notification_type` field
# — the listener would drop it. We translate it here into a daemon event
# whose `text` is pre-baked, so synthesis skips the LLM (the daemon already
# special-cases `text=...` for the manual `jarvis-cc say --text` path).
#
# Output language is governed by `behavior.voice_language` ("en" | "zh" |
# "auto"). Default is "en" — the user's chosen British voice identity.
# "auto" picks per-event from the question text (CJK → zh).
_EN_ORDINALS = ("Option one", "Option two", "Option three", "Option four")
_ZH_ORDINALS = ("选项一", "选项二", "选项三", "选项四")


def _has_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


def _resolve_lang(mode: str, questions: list) -> str:
    """Pick output lang from the user's mode setting and question content."""
    if mode == "zh":
        return "zh"
    if mode == "auto":
        first_q = ""
        if questions and isinstance(questions[0], dict):
            first_q = questions[0].get("question") or ""
        return "zh" if _has_cjk(first_q) else "en"
    return "en"  # "en" and any unrecognized value


def _render_askuserquestion(questions: list, lang: str) -> str | None:
    if not questions or not isinstance(questions[0], dict):
        return None
    first = questions[0]
    q_text = (first.get("question") or "").strip()
    options = first.get("options") or []
    if not q_text or not options:
        return None

    q_clean = q_text.rstrip("。.!?！？")
    if lang == "zh":
        ordinals = _ZH_ORDINALS
        head = q_clean if q_clean.startswith(("先生", "Sir,", "Sir ")) else "先生，" + q_clean
        sep = "。"
        remaining_tmpl = "另有 {n} 个问题等候，先生。"
    else:
        ordinals = _EN_ORDINALS
        head = (
            q_clean
            if q_clean.lower().startswith(("sir,", "sir "))
            else "Sir, " + q_clean
        )
        sep = ". "
        remaining_tmpl = "And {n} more questions on screen, sir."

    parts = [head + sep]
    for i, opt in enumerate(options[:4]):
        if not isinstance(opt, dict):
            continue
        label = (opt.get("label") or "").strip()
        if not label:
            continue
        parts.append(f"{ordinals[i]}: {label}{sep}")

    remaining = len(questions) - 1
    if remaining > 0:
        parts.append(remaining_tmpl.format(n=remaining))
    return "".join(parts).rstrip()


def _first_question_is_usable(questions: list) -> bool:
    if not questions or not isinstance(questions[0], dict):
        return False
    first = questions[0]
    q_text = (first.get("question") or "").strip()
    options = first.get("options") or []
    return bool(q_text and options)


def _translate_cc_payload(payload: dict, lang_mode: str = "en") -> dict | None:
    """Rewrite a raw hook payload (Claude Code OR Codex CLI) into the
    daemon's normalized shape.

    Both products send lifecycle-hook payloads with the same snake_case
    shape and overlapping `hook_event_name` vocabulary (UserPromptSubmit,
    PostToolUse, PreToolUse), so those code paths are shared. Codex-only
    pieces handled here:
      * `PermissionRequest` → translated to a permission_prompt event so
        the daemon's existing routing pipeline can speak it.
      * `notify`'s flat `agent-turn-complete` payload (kebab-case, no
        `hook_event_name`) → translated to idle_prompt.

    Returns the new dict, or None if the payload can't/shouldn't be
    forwarded. Returns the original (unchanged) for payloads already in
    daemon shape.

    For AskUserQuestion the behavior depends on `lang_mode`:
      - "auto" → render verbatim text + lang in the hook (no LLM round-trip);
        CJK question text picks zh, otherwise en.
      - "en" / "zh" → forward the questions WITHOUT text/lang so the daemon's
        phrase router calls Ollama/DeepSeek to translate-and-rephrase into a
        Jarvis-toned line in the user's chosen output language.
    """
    # Codex `notify` payload — flat, kebab-case keys, `type` discriminator.
    # Currently only `agent-turn-complete` is emitted, but check by type
    # rather than presence so future Codex notify types fall through to
    # a no-op rather than triggering the wrong event.
    if payload.get("type") == "agent-turn-complete":
        return {
            "notification_type": "idle_prompt",
            "tool_name": None,
            "tool_input": {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("thread-id"),
        }

    hook_event = payload.get("hook_event_name")
    if hook_event in ("UserPromptSubmit", "PostToolUse"):
        sid = payload.get("session_id")
        if not sid:
            return None
        return {"command": "cancel", "session_id": sid}
    # Codex permission prompt — no Claude Code analogue. Translate to
    # the daemon's permission_prompt notification so the existing phrase
    # router + TTS pipeline handles it identically.
    if hook_event == "PermissionRequest":
        return {
            "notification_type": "permission_prompt",
            "tool_name": payload.get("tool_name"),
            "tool_input": payload.get("tool_input") or {},
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }
    if hook_event == "PreToolUse" and \
            payload.get("tool_name") == "AskUserQuestion":
        ti = payload.get("tool_input") or {}
        questions = ti.get("questions")
        if not isinstance(questions, list) or not _first_question_is_usable(questions):
            return None

        base: dict = {
            "notification_type": "ask_user_question",
            "tool_name": "AskUserQuestion",
            "cwd": payload.get("cwd"),
            "session_id": payload.get("session_id"),
        }

        if lang_mode == "auto":
            lang = _resolve_lang("auto", questions)
            text = _render_askuserquestion(questions, lang)
            if not text:
                return None
            return {**base, "tool_input": {}, "text": text, "lang": lang}

        # "en" / "zh" / unknown: forward verbatim questions so the daemon's
        # phrase router can translate them into the configured voice_language.
        return {**base, "tool_input": {"questions": questions}}
    return payload


def forward_event(
    stream: IO[str],
    sock_path: str | Path,
    *,
    lang_mode: str = "en",
    cancel_on_user_action: bool = True,
) -> bool:
    """Forward an NDJSON event from `stream` to the unix socket at `sock_path`.

    `lang_mode` ("en" | "zh" | "auto") only affects AskUserQuestion translation.
    `cancel_on_user_action`: when False, UserPromptSubmit / PostToolUse hook
    payloads are dropped without contacting the daemon.

    Returns True if successfully sent. Returns False on any failure
    (invalid JSON, socket missing, write error, dropped by policy) —
    never raises.
    """
    sock_path = Path(sock_path)
    try:
        raw = stream.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return False

    if not isinstance(payload, dict):
        return False

    if (
        not cancel_on_user_action
        and payload.get("hook_event_name") in ("UserPromptSubmit", "PostToolUse")
    ):
        return False

    payload = _translate_cc_payload(payload, lang_mode=lang_mode)
    if payload is None:
        return False

    payload["_received_at"] = time.time()
    line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(str(sock_path))
        s.sendall(line)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def main() -> int:
    """Entry point registered as `jarvis-cc-hook` console_script.

    Must NEVER raise — Claude Code reads stdout for hook decisions and a
    traceback would corrupt that channel. All failures are silent and
    exit 0.
    """
    try:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        mode = getattr(cfg.behavior, "voice_language", "en") or "en"
        cancel = getattr(cfg.behavior, "cancel_on_user_action", True)
        forward_event(
            sys.stdin,
            cfg.paths.socket,
            lang_mode=mode,
            cancel_on_user_action=cancel,
        )
    except Exception:  # noqa: BLE001 — structural guarantee
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
