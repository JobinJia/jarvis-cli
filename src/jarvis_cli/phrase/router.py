"""Provider routing: try primary, then fallback, then template.

Owns the extract → redact → build_messages pipeline so providers stay
dumb HTTP adapters that just take pre-built messages and return a string.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

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
        *,
        on_primary_fallback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cfg = cfg
        # Fires exactly when the primary provider failed/empty AND the
        # fallback then produced a usable phrase — i.e. the moment a
        # local-first deployment quietly slipped onto the cloud path.
        # Callers wire this up to surface "Sir, switching to cloud" so
        # the user notices before the meter ticks for a while unnoticed.
        self._on_primary_fallback = on_primary_fallback

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
            humor_level=self.cfg.behavior.humor_level,
        )
        primary_failed = False
        for i, provider in enumerate((self.primary, self.fallback)):
            if provider is None:
                continue
            try:
                out = await provider.generate(messages)
                if out and out.strip():
                    if (
                        i == 1
                        and primary_failed
                        and self._on_primary_fallback is not None
                        and self.primary is not None
                    ):
                        try:
                            await self._on_primary_fallback(self.primary.name)
                        except Exception as cb_exc:  # noqa: BLE001
                            logger.warning(
                                "on_primary_fallback callback failed: {}", cb_exc
                            )
                    return out.strip()
                # Empty output counts as a failure for fallback-trigger purposes.
                if i == 0:
                    primary_failed = True
            except Exception as exc:
                if i == 0:
                    primary_failed = True
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
