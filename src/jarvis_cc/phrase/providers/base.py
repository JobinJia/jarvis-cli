"""Abstract base for LLM phrase providers."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ...types import Event, Lang


class PhraseProvider(ABC):
    """A provider returns a single Jarvis-tone sentence for an Event."""

    name: str

    @abstractmethod
    async def generate(self, event: Event, lang: Lang, max_chars: int) -> str: ...

    async def healthcheck(self) -> bool:
        """Return True if this provider is likely to succeed right now."""
        return True
