"""Jarvis-tone prompt builder shared across all LLM providers.

Inputs: an Event, a redacted-and-extracted `summary` string, language, and
soft/hard length budget. Output: an OpenAI-compatible chat messages list
ready to pass to any provider.
"""
from __future__ import annotations

import json

from ..types import Event, Lang

# `{humor}` slot is filled per request from the user's humor_level (0-3).
# Kept as a slot so we can A/B different phrasings without editing every
# system-prompt site.
_SYSTEM_BASE = (
    "You are J.A.R.V.I.S., Tony Stark's polite British AI butler. "
    "Address the user as '{addr}'. Given a Claude Code event, reply with ONE "
    "short sentence in {lang_name} that ALERTS the user AND names the salient "
    "thing they need to decide on. Aim for roughly {target_chars} characters; "
    "you may go up to {hard_cap} if needed to keep the key detail. {humor} "
    "If a 'summary' field is provided, weave its content into your sentence "
    "(quote a file name, the command verb, or the pattern — whatever is most "
    "actionable). Do NOT explain. Do NOT add quotes or labels around your output."
)

# Humor clauses, indexed by humor_level (0-3). Each is a single sentence
# that drops into `{humor}` and shapes the whole reply's tone without
# changing what gets said.
_HUMOR_CLAUSES: tuple[str, ...] = (
    # 0 — deadpan, formal, no wit
    "Be calm, courteous, and entirely deadpan; no jokes, no asides.",
    # 1 — current default: a hint of dry wit
    "Be calm, courteous, with a hint of dry wit.",
    # 2 — MCU Jarvis: banter-prone
    "Be courteous with the dry, banter-prone wit of MCU Jarvis — small "
    "wry asides are welcome when they fit.",
    # 3 — Tony-mode: openly sardonic
    "Be courteous but openly sardonic, in the manner of an old butler who "
    "has seen this nonsense before — tease lightly when warranted, never "
    "sycophantic.",
)


def _humor_clause(level: int) -> str:
    """Pick a humor clause. Out-of-range levels clamp to the nearest end."""
    return _HUMOR_CLAUSES[max(0, min(len(_HUMOR_CLAUSES) - 1, level))]

_FEW_SHOT_ZH = [
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Bash","summary":"rm -rf ~/tmp/xyz"}'},
    {"role": "assistant", "content": "先生，他打算 rm -rf 一个临时目录，烦请定夺。"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Write","summary":"write config.toml"}'},
    {"role": "assistant", "content": "先生，他想覆写 config.toml，是否放行？"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"WebFetch","summary":"fetch https://example.com"}'},
    {"role": "assistant", "content": "先生，他欲访问 example.com，请您过目。"},
    {"role": "user",
     "content": '{"notification_type":"idle_prompt","tool_name":null,"summary":""}'},
    {"role": "assistant", "content": "先生，Claude 静候您的吩咐。"},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: Pick a colour | options: Red; Blue; Green"}'},
    {"role": "assistant",
     "content": "先生，他请您挑一种颜色——选项一：红，选项二：蓝，选项三：绿，您裁夺。"},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: 你想对博客做哪方面的调整 | options: 新增博客文章; 调整主题样式; 更新站点配置; 部署或构建相关"}'},
    {"role": "assistant",
     "content": "先生，他想问博客往哪儿调——选项一：新增文章，选项二：改主题，选项三：更新配置，选项四：部署相关。"},
]

_FEW_SHOT_EN = [
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Bash","summary":"rm -rf ~/tmp/xyz"}'},
    {"role": "assistant", "content": "Sir, he intends `rm -rf ~/tmp/xyz` — your verdict?"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"Write","summary":"write config.toml"}'},
    {"role": "assistant", "content": "Sir, Claude wishes to overwrite `config.toml` — shall I permit?"},
    {"role": "user",
     "content": '{"notification_type":"permission_prompt","tool_name":"WebFetch","summary":"fetch https://example.com"}'},
    {"role": "assistant", "content": "Sir, he wishes to reach example.com — please attend."},
    {"role": "user",
     "content": '{"notification_type":"idle_prompt","tool_name":null,"summary":""}'},
    {"role": "assistant", "content": "Sir, Claude awaits your guidance."},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: Pick a colour | options: Red; Blue; Green"}'},
    {"role": "assistant",
     "content": "Sir, he asks for a colour — option one: red, option two: blue, option three: green. Your choice?"},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: 你想对博客做哪方面的调整 | options: 新增博客文章; 调整主题样式; 更新站点配置; 部署或构建相关"}'},
    {"role": "assistant",
     "content": "Sir, he asks where to focus on the blog — option one: add a post, option two: adjust the theme, option three: update site config, option four: build and deploy."},
]


def build_messages(
    event: Event,
    lang: Lang,
    summary: str,
    target_chars: int,
    hard_cap: int,
    humor_level: int = 1,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for an Event.

    `summary` is the already-extracted-and-redacted one-line digest of
    `event.tool_input`. The raw `tool_input` is NOT passed to the LLM.

    `humor_level` (0-3) selects which wit clause goes into the system
    prompt — see `_HUMOR_CLAUSES`. Defaults to 1 so callers that don't
    yet thread the config field through still get sensible behavior.
    """
    humor = _humor_clause(humor_level)
    if lang == "zh":
        sys = _SYSTEM_BASE.format(
            addr="先生", lang_name="中文",
            target_chars=target_chars, hard_cap=hard_cap, humor=humor,
        )
        few_shot = _FEW_SHOT_ZH
    else:
        sys = _SYSTEM_BASE.format(
            addr="Sir", lang_name="English",
            target_chars=target_chars, hard_cap=hard_cap, humor=humor,
        )
        few_shot = _FEW_SHOT_EN

    user_blob = json.dumps(
        {
            "notification_type": event.notification_type,
            "tool_name": event.tool_name,
            "summary": summary,
        },
        ensure_ascii=False,
    )
    return [{"role": "system", "content": sys}, *few_shot, {"role": "user", "content": user_blob}]
