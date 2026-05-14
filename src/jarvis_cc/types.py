"""Shared dataclasses used across hook_client, daemon, phrase, tts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NotificationType = Literal[
    "permission_prompt",
    "idle_prompt",
    "elicitation_dialog",
]

Lang = Literal["zh", "en"]


@dataclass(frozen=True)
class Event:
    """A single notification event from Claude Code, normalized for daemon use."""

    notification_type: NotificationType
    tool_name: str | None
    tool_input: dict[str, Any] = field(default_factory=dict)
    cwd: str | None = None
    session_id: str | None = None
    raw_message: str | None = None
    received_at: float = 0.0  # epoch seconds; filled by listener
    # Pre-baked text from the caller. When set the daemon SKIPS the phrase
    # router (no LLM round-trip) and synthesizes this string verbatim. Used
    # by `jarvis-cc say --text` for assistant-side scenarios CC doesn't cover.
    text: str | None = None
    lang: Lang | None = None  # only honored when `text` is set
    voice_id: str | None = None  # per-event TTS voice override (eg EL voice_id)

    def dedup_key(self) -> str:
        """Hash key for dedup window: same (cwd, type, tool) collapses."""
        return f"{self.cwd or ''}::{self.notification_type}::{self.tool_name or ''}"
