"""Provider routing: try primary, then fallback, then template.

Owns the extract → redact → build_messages pipeline so providers stay
dumb HTTP adapters that just take pre-built messages and return a string.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from loguru import logger

from ..config import Config
from ..types import Emotion, Event, Lang
from . import extract, redact
from .chunker import chunk_sentences
from .prompt import build_messages
from .providers.base import PhraseProvider
from .templates import render_template


class PhraseRouter:
    def __init__(
        self,
        primary: PhraseProvider | None,
        fallback: PhraseProvider | None = None,
        cfg: Config | None = None,
        *,
        fallbacks: list[PhraseProvider] | None = None,
        on_primary_fallback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.primary = primary
        # Fallback chain: each is tried in order after the primary, so one free
        # cloud provider being rate-limited (e.g. Zhipu 1305) falls through to
        # the next instead of straight to the template. `fallbacks` wins; the
        # singular `fallback` arg is kept for back-compat (wrapped into a list).
        self.fallbacks: list[PhraseProvider] = (
            list(fallbacks)
            if fallbacks is not None
            else ([fallback] if fallback is not None else [])
        )
        self.cfg = cfg
        # Fires exactly when the primary provider failed/empty AND the
        # fallback then produced a usable phrase — i.e. the moment a
        # local-first deployment quietly slipped onto the cloud path.
        # Callers wire this up to surface "Sir, switching to cloud" so
        # the user notices before the meter ticks for a while unnoticed.
        self._on_primary_fallback = on_primary_fallback
        # Last spoken idle line, threaded into the next idle request as
        # `avoid` so the model doesn't repeat itself back-to-back.
        self._last_idle_line: str | None = None

    def _address_for(self, lang: Lang) -> str | None:
        """The configured form of address for `lang`, or None (prompt default)
        when no config is attached."""
        if self.cfg is None:
            return None
        return (
            self.cfg.behavior.address_zh
            if lang == "zh"
            else self.cfg.behavior.address_en
        )

    async def phrase(
        self, event: Event, lang: Lang,
        emotion: Emotion | None = None,
    ) -> str:
        text = await self._phrase_inner(event, lang, emotion=emotion)
        if event.notification_type == "idle_prompt":
            self._last_idle_line = text
        return text

    async def _phrase_inner(
        self, event: Event, lang: Lang,
        emotion: Emotion | None = None,
    ) -> str:
        if event.notification_type == "tool_failure":
            summary = extract.extract_failure(event.tool_name, event.tool_input)
        elif event.notification_type == "api_error":
            summary = extract.extract_api_error(event.tool_input)
        else:
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
            avoid=self._last_idle_line
            if event.notification_type == "idle_prompt" else None,
            emotion=emotion,
            address=self._address_for(lang),
        )
        primary_failed = False
        for i, provider in enumerate([self.primary, *self.fallbacks]):
            if provider is None:
                continue
            try:
                out = await provider.generate(messages)
                if out and out.strip():
                    # Any fallback (i > 0) producing a phrase after the primary
                    # failed means we quietly slipped onto a cloud path — fire
                    # the alert once (only on the level that actually answers).
                    if (
                        i > 0
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

    async def phrase_stream(
        self, event: Event, lang: Lang,
        emotion: Emotion | None = None,
    ) -> AsyncIterator[str]:
        """Stream sentences as the LLM produces tokens.

        Yields complete sentences (via the chunker) from the primary
        provider's ``generate_stream()``.  On any streaming failure, falls
        back to the non-streaming ``phrase()`` and yields the whole result
        as a single chunk — callers get at least one yield in all cases.
        """
        try:
            async for sentence in self._phrase_stream_inner(
                event, lang, emotion=emotion,
            ):
                yield sentence
        except Exception as exc:
            logger.warning("Streaming phrase failed ({}), falling back to batch", exc)
            text = await self.phrase(event, lang, emotion=emotion)
            yield text

    async def _phrase_stream_inner(
        self, event: Event, lang: Lang,
        emotion: Emotion | None = None,
    ) -> AsyncIterator[str]:
        """Build messages and stream sentences from the primary provider."""
        if event.notification_type == "tool_failure":
            summary = extract.extract_failure(event.tool_name, event.tool_input)
        else:
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
            avoid=self._last_idle_line
            if event.notification_type == "idle_prompt" else None,
            emotion=emotion,
            address=self._address_for(lang),
        )
        if self.primary is None:
            raise RuntimeError("No primary provider configured")
        token_stream = self.primary.generate_stream(messages)
        collected: list[str] = []
        async for sentence in chunk_sentences(token_stream):
            collected.append(sentence)
            yield sentence
        # Update last-idle state from the full text so the non-streaming
        # dedup logic stays consistent.
        if event.notification_type == "idle_prompt" and collected:
            self._last_idle_line = " ".join(collected)
