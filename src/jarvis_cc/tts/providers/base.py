"""Abstract base for TTS providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path

from ...types import Lang


class TTSProvider(ABC):
    """Synthesize `text` in `lang` to an audio file at `out_path` and return it.

    `voice_id` is an optional per-call override. Providers that have no notion
    of a swappable voice (eg macOS `say`) ignore it.

    Providers that can deliver audio bytes incrementally set
    `supports_streaming = True` and implement `stream()`; the daemon then
    plays audio as bytes arrive rather than waiting for full synthesis.
    """

    name: str
    supports_streaming: bool = False

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
    ) -> Path: ...

    async def stream(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield audio bytes incrementally. Default impl signals 'not supported'."""
        raise NotImplementedError(f"{self.name} does not support streaming")
        # Make the body an async generator for type-checkers.
        if False:
            yield b""

    async def healthcheck(self) -> bool:
        return True
