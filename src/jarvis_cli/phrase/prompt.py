"""Jarvis-tone prompt builder shared across all LLM providers.

Inputs: an Event, a redacted-and-extracted `summary` string, language, and
soft/hard length budget. Output: an OpenAI-compatible chat messages list
ready to pass to any provider.
"""
from __future__ import annotations

import json
import random

from ..types import Emotion, Event, Lang, emotion_for

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


# Improvisation angles for idle_prompt. The model writes the sentence; these
# only point it somewhere new each time. Without a varying input, small
# models converge on one or two stock phrasings regardless of temperature.
_IDLE_FLAVORS: tuple[str, ...] = (
    "a chess move awaited",
    "a theatre stage gone quiet",
    "a military situation report",
    "a ship's bridge awaiting orders",
    "an orchestra awaiting its conductor",
    "a butler announcing that everything is ready",
    "a race pit crew on standby",
    "the calm before the storm",
    "a telegraph line gone silent",
    "a chauffeur idling the engine",
    "a library hush",
    "tea served and going cold",
)

# Unified emotion clauses: each emotion maps to a tone instruction appended
# to the system prompt. Replaces the old per-event _FAILURE_CLAUSE /
# _COMPLETE_CLAUSE approach — the event type now selects an emotion (via
# EVENT_EMOTION in types.py), and the emotion selects the clause here.
_EMOTION_CLAUSES: dict[str, str] = {
    "warm": "Speak warmly and welcomingly, as though greeting a returning friend.",
    "neutral": "",
    "gentle": "Speak gently and patiently, with a soft invitation.",
    "grave": "Speak gravely and concisely — state what failed and why. No banter.",
    "pleased": "Speak with quiet satisfaction, brief and light.",
    "sardonic": "Speak with sardonic amusement, as though unsurprised by the absurdity.",
}

# Appended to the system prompt only for idle_prompt requests.
_IDLE_CLAUSE = (
    " This is an idle notification with nothing to report; improvise a fresh "
    "one-liner inviting the user back, riffing on the 'flavor' hint. Never "
    "reuse the phrasing in 'avoid', and do not copy the example reply."
)

# Appended for tool_failure: behavioral frame on top of the emotion clause.
_FAILURE_CLAUSE = (
    " This is a FAILURE report: a tool or command just failed. State what "
    "failed and the gist of why, in ONE short sentence. No reassurance, "
    "no banter, no jokes."
)

# Appended for task_complete: behavioral frame on top of the emotion clause.
_COMPLETE_CLAUSE = (
    " This is a COMPLETION notice: Claude just finished responding. Reply with "
    "a very brief acknowledgement of three to six words (e.g. 'All done, sir.'). "
    "Do not summarise the work; do not ask a question."
)

# --- Tier 1 lifecycle clauses ---

# Context about to be compressed (PreCompact). A brief heads-up.
_COMPACT_CLAUSE = (
    " This is a CONTEXT-COMPACTION alert: the conversation context is about to "
    "be compressed. State this calmly in ONE short sentence. No details needed."
)

# Rate-limit hit (RateLimitError). Keep it matter-of-fact.
_RATE_LIMIT_CLAUSE = (
    " This is a RATE-LIMIT alert: Claude has hit the API rate limit and must "
    "pause briefly. Announce the pause calmly in ONE short sentence."
)

# Sub-agent dispatched (SubagentStart). Brief acknowledgement.
_SUBAGENT_CLAUSE = (
    " This is a SUB-AGENT dispatch notice: a sub-agent has been spawned. "
    "Announce it briefly in ONE short sentence."
)

# Turn limit reached (MaxTurnsReached). Claude has stopped.
_MAX_TURNS_CLAUSE = (
    " This is a TURN-LIMIT notice: Claude has reached its maximum number of "
    "turns and stopped. State this gravely in ONE short sentence."
)

# --- Tier 2 lifecycle clauses ---

# API error (APIError). Grave, concise — like tool_failure.
_API_ERROR_CLAUSE = (
    " This is an API ERROR report: the API returned an error. Speak gravely "
    "and concisely — state the gist of the error in ONE short sentence. "
    "No reassurance, no banter."
)

# Session ended (SessionStop). A farewell.
_SESSION_END_CLAUSE = (
    " This is a SESSION-END notice: the Claude session is ending. Reply with "
    "a brief, warm farewell of three to six words. Do not summarise the work."
)

# Context compacted (PostCompact). Brief 'done, moving on'.
_POST_COMPACT_CLAUSE = (
    " This is a POST-COMPACTION notice: the context has been compressed. "
    "Reply with a very brief acknowledgement of a few words indicating "
    "compression is done and work continues."
)

# Context window full (ContextWindowOverflow). Urgent alert.
_CONTEXT_OVERFLOW_CLAUSE = (
    " This is a CONTEXT-OVERFLOW alert: the context window is completely full. "
    "State this urgently in ONE short sentence. No reassurance."
)


def _pick_flavor() -> str:
    return random.choice(_IDLE_FLAVORS)

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
     "content": '{"notification_type":"idle_prompt","tool_name":null,"summary":"",'
                '"flavor":"an orchestra awaiting its conductor","avoid":"先生，Claude 静候您的吩咐。"}'},
    {"role": "assistant", "content": "先生，乐团已就位，只候您执棒。"},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: Pick a colour | options: Red; Blue; Green"}'},
    {"role": "assistant",
     "content": "先生，他请您挑一种颜色——选项一：红，选项二：蓝，选项三：绿，您裁夺。"},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: 你想对博客做哪方面的调整 | options: 新增博客文章; 调整主题样式; 更新站点配置; 部署或构建相关"}'},
    {"role": "assistant",
     "content": "先生，他想问博客往哪儿调——选项一：新增文章，选项二：改主题，选项三：更新配置，选项四：部署相关。"},
    {"role": "user",
     "content": '{"notification_type":"tool_failure","tool_name":"Bash","summary":"Bash failed: npm test: 3 tests failed"}'},
    {"role": "assistant", "content": "先生，测试未通过——三个用例失败了。"},
    {"role": "user",
     "content": '{"notification_type":"task_complete","tool_name":null,"summary":""}'},
    {"role": "assistant", "content": "全部办妥，先生。"},
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
     "content": '{"notification_type":"idle_prompt","tool_name":null,"summary":"",'
                '"flavor":"a chess move awaited","avoid":"Sir, Claude awaits your guidance."}'},
    {"role": "assistant", "content": "Your move, sir — the board is set."},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: Pick a colour | options: Red; Blue; Green"}'},
    {"role": "assistant",
     "content": "Sir, he asks for a colour — option one: red, option two: blue, option three: green. Your choice?"},
    {"role": "user",
     "content": '{"notification_type":"ask_user_question","tool_name":"AskUserQuestion","summary":"ask: 你想对博客做哪方面的调整 | options: 新增博客文章; 调整主题样式; 更新站点配置; 部署或构建相关"}'},
    {"role": "assistant",
     "content": "Sir, he asks where to focus on the blog — option one: add a post, option two: adjust the theme, option three: update site config, option four: build and deploy."},
    {"role": "user",
     "content": '{"notification_type":"tool_failure","tool_name":"Bash","summary":"Bash failed: npm test: 3 tests failed"}'},
    {"role": "assistant", "content": "Sir, the build failed — three tests did not pass."},
    {"role": "user",
     "content": '{"notification_type":"task_complete","tool_name":null,"summary":""}'},
    {"role": "assistant", "content": "All done, sir."},
]


def build_messages(
    event: Event,
    lang: Lang,
    summary: str,
    target_chars: int,
    hard_cap: int,
    humor_level: int = 1,
    avoid: str | None = None,
    emotion: Emotion | None = None,
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for an Event.

    `summary` is the already-extracted-and-redacted one-line digest of
    `event.tool_input`. The raw `tool_input` is NOT passed to the LLM.

    `humor_level` (0-3) selects which wit clause goes into the system
    prompt — see `_HUMOR_CLAUSES`. Defaults to 1 so callers that don't
    yet thread the config field through still get sensible behavior.

    `avoid` (idle_prompt only) is the previously spoken idle line; the
    request carries it plus a random `flavor` hint so the model improvises
    a fresh sentence instead of parroting one stock phrase.

    `emotion` selects a tone clause from `_EMOTION_CLAUSES`. When None,
    the emotion is derived from the event's notification_type via
    `emotion_for()`.
    """
    humor = _humor_clause(humor_level)
    emo = emotion or emotion_for(event.notification_type)
    is_idle = event.notification_type == "idle_prompt"
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

    # Inject the emotion clause (warm, grave, pleased, etc.) — shapes the
    # LLM's written text toward the target tone for ALL TTS providers, even
    # those that have no synthesis-level emotion knobs.
    emo_clause = _EMOTION_CLAUSES.get(emo, "")
    if emo_clause:
        sys += " " + emo_clause

    # Per-event behavioral frame (idle flavor, failure/completion framing)
    # layered on top of the emotion clause.
    if is_idle:
        sys += _IDLE_CLAUSE
    elif event.notification_type == "tool_failure":
        sys += _FAILURE_CLAUSE
    elif event.notification_type == "task_complete":
        sys += _COMPLETE_CLAUSE
    elif event.notification_type == "context_compacting":
        sys += _COMPACT_CLAUSE
    elif event.notification_type == "rate_limited":
        sys += _RATE_LIMIT_CLAUSE
    elif event.notification_type == "subagent_spawned":
        sys += _SUBAGENT_CLAUSE
    elif event.notification_type == "max_turns_reached":
        sys += _MAX_TURNS_CLAUSE
    elif event.notification_type == "api_error":
        sys += _API_ERROR_CLAUSE
    elif event.notification_type == "session_end":
        sys += _SESSION_END_CLAUSE
    elif event.notification_type == "context_compacted":
        sys += _POST_COMPACT_CLAUSE
    elif event.notification_type == "context_overflow":
        sys += _CONTEXT_OVERFLOW_CLAUSE

    blob: dict[str, object] = {
        "notification_type": event.notification_type,
        "tool_name": event.tool_name,
        "summary": summary,
    }
    if is_idle:
        blob["flavor"] = _pick_flavor()
        if avoid:
            blob["avoid"] = avoid
    user_blob = json.dumps(blob, ensure_ascii=False)
    return [{"role": "system", "content": sys}, *few_shot, {"role": "user", "content": user_blob}]
