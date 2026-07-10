"""TTS engine: chains primary → fallback providers, with per-language routing."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from ..types import Lang
from .providers.base import TTSProvider


class TTSEngine:
    def __init__(
        self,
        primary: TTSProvider,
        fallback: TTSProvider | None,
        overrides: dict[Lang, TTSProvider] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        # Per-language primary override (tts.provider_zh). Why: one provider
        # rarely speaks both languages natively — the XTTS Bettany clone is
        # English-born and reads Chinese with a foreign accent, while a fixed
        # zh voice has no business reading the English lines. Routing picks
        # the native speaker per utterance; the fallback chain is shared.
        self.overrides: dict[Lang, TTSProvider] = overrides or {}

    def primary_for(self, lang: Lang) -> TTSProvider:
        """The primary provider for `lang` — the override when one is
        configured, else the global primary."""
        return self.overrides.get(lang, self.primary)

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> Path:
        for provider in (self.primary_for(lang), self.fallback):
            if provider is None:
                continue
            try:
                return await provider.synthesize(
                    text, lang, out_path,
                    voice_id=voice_id, emotion=emotion,
                )
            except Exception as exc:
                logger.warning("TTS provider {} failed: {}", provider.name, exc)
        raise RuntimeError("All TTS providers failed")
