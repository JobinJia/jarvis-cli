"""Abstract base for LLM phrase providers."""
from __future__ import annotations

from abc import ABC, abstractmethod


class PhraseProvider(ABC):
    """A provider returns a single Jarvis-tone sentence given pre-built
    OpenAI-compatible chat `messages`. The router is responsible for
    constructing `messages` (extract + redact + build_messages); providers
    are dumb HTTP adapters.
    """

    name: str

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str: ...

    async def healthcheck(self) -> bool:
        return True
