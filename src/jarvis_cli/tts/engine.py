"""TTS engine: chains primary → fallback providers."""
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
    ) -> None:
        self.primary = primary
        self.fallback = fallback

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> Path:
        for provider in (self.primary, self.fallback):
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
