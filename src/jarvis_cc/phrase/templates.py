"""Fallback phrase templates when all LLM providers fail.

Tone tries to approximate Jarvis: polite, brief, with "Sir"/"先生".
"""
from __future__ import annotations

from ..types import Event, Lang

_ZH: dict[str, str] = {
    "permission_prompt": "先生，Claude 请求使用 {tool} 的权限。",
    "idle_prompt": "先生，Claude 正在等候您的指示。",
    "elicitation_dialog": "先生，有个对话框等您填写。",
}

_EN: dict[str, str] = {
    "permission_prompt": "Sir, Claude requests permission for {tool}.",
    "idle_prompt": "Sir, Claude awaits your guidance.",
    "elicitation_dialog": "Sir, a dialog awaits your input.",
}


def render_template(event: Event, lang: Lang) -> str:
    table = _ZH if lang == "zh" else _EN
    tmpl = table.get(event.notification_type, table["idle_prompt"])
    return tmpl.format(tool=event.tool_name or "something")
