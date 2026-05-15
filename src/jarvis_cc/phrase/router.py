"""Provider routing: try primary, then fallback, then template.

Owns the extract → redact → build_messages pipeline so providers stay
dumb HTTP adapters that just take pre-built messages and return a string.
"""
from __future__ import annotations

from loguru import logger

from ..config import Config
from ..types import Event, Lang
from . import extract, redact
from .prompt import build_messages
from .providers.base import PhraseProvider
from .templates import render_template


class PhraseRouter:
    def __init__(
        self,
        primary: PhraseProvider | None,
        fallback: PhraseProvider | None,
        cfg: Config,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cfg = cfg

    async def phrase(self, event: Event, lang: Lang) -> str:
        summary = extract.extract(event.tool_name, event.tool_input)
        summary = redact.scrub(
            summary,
            enabled=self.cfg.behavior.privacy.cloud_redaction,
        )
        target_chars, hard_cap = self._budget_for(event)
        messages = build_messages(
            event, lang, summary,
            target_chars=target_chars,
            hard_cap=hard_cap,
        )
        for provider in (self.primary, self.fallback):
            if provider is None:
                continue
            try:
                out = await provider.generate(messages)
                if out and out.strip():
                    return out.strip()
            except Exception as exc:
                logger.warning(
                    "Phrase provider {} failed: {}", provider.name, exc
                )
        return render_template(event, lang)

    def _budget_for(self, event: Event) -> tuple[int, int]:
        # AskUserQuestion is intrinsically longer (question + up to 4 option
        # labels) than other events; give it more room than the default
        # phrase budget so the LLM can enumerate options without truncation.
        if event.notification_type == "ask_user_question":
            return 200, 400
        return (
            self.cfg.behavior.phrase_target_chars,
            self.cfg.behavior.phrase_hard_cap,
        )
