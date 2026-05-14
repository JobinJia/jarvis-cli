"""Jarvis-tone prompt builder shared across all LLM providers."""
from __future__ import annotations

import json

from ..types import Event, Lang

_SYSTEM_BASE = (
    "You are J.A.R.V.I.S., Tony Stark's polite British AI butler. "
    "Address the user as '{addr}'. "
    "Given a structured Claude Code event, reply with a single short sentence "
    "in {lang_name} (at most {max_chars} characters), notifying the user of "
    "what needs their decision. Be calm, courteous, with a hint of dry wit. "
    "Do NOT explain reasoning. Do NOT add quotes or labels. Output the sentence only."
)

_FEW_SHOT_ZH = [
    {"role": "user", "content": '{"notification_type":"permission_prompt","tool_name":"Bash"}'},
    {"role": "assistant", "content": "先生，Claude 想动用一下终端，请您过目。"},
    {"role": "user", "content": '{"notification_type":"idle_prompt","tool_name":null}'},
    {"role": "assistant", "content": "先生，Claude 静候您的吩咐。"},
]

_FEW_SHOT_EN = [
    {"role": "user", "content": '{"notification_type":"permission_prompt","tool_name":"Bash"}'},
    {"role": "assistant", "content": "Sir, Claude requests the shell, at your discretion."},
    {"role": "user", "content": '{"notification_type":"idle_prompt","tool_name":null}'},
    {"role": "assistant", "content": "Sir, Claude awaits your guidance."},
]


def build_messages(event: Event, lang: Lang, max_chars: int) -> list[dict[str, str]]:
    """Return the OpenAI-compatible chat messages for an Event."""
    if lang == "zh":
        sys = _SYSTEM_BASE.format(addr="先生", lang_name="中文", max_chars=max_chars)
        few_shot = _FEW_SHOT_ZH
    else:
        sys = _SYSTEM_BASE.format(addr="Sir", lang_name="English", max_chars=max_chars)
        few_shot = _FEW_SHOT_EN

    user_blob = json.dumps(
        {
            "notification_type": event.notification_type,
            "tool_name": event.tool_name,
            "tool_input": event.tool_input,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": sys}, *few_shot, {"role": "user", "content": user_blob}]
