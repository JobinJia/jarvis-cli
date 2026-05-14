"""Provider routing: try primary, then fallback, then template."""
from __future__ import annotations

from loguru import logger

from ..config import Config
from ..types import Event, Lang
from .providers.base import PhraseProvider
from .templates import render_template


class PhraseRouter:
    """Owns the LLM provider chain. Always returns a string."""

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
        max_chars = self.cfg.behavior.phrase_max_chars
        for provider in (self.primary, self.fallback):
            if provider is None:
                continue
            try:
                out = await provider.generate(event, lang, max_chars)
                if out and out.strip():
                    return out.strip()
            except Exception as exc:  # broad: any provider failure → next
                logger.warning(
                    "Phrase provider {} failed: {}", provider.name, exc
                )
        return render_template(event, lang)
