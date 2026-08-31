"""Abstract base for LLM phrase providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class PhraseProvider(ABC):
    """A provider returns a single Jarvis-tone sentence given pre-built
    OpenAI-compatible chat `messages`. The router is responsible for
    constructing `messages` (extract + redact + build_messages); providers
    are dumb HTTP adapters.
    """

    name: str

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str: ...

    async def generate_stream(
        self, messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Yield token deltas as they arrive from the LLM.

        Default implementation calls the non-streaming ``generate()`` and
        yields the whole result as a single chunk — providers that support
        server-sent events override this to yield incremental tokens.
        """
        text = await self.generate(messages)
        yield text

    async def healthcheck(self) -> bool:
        return True
