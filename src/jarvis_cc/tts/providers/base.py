"""Abstract base for TTS providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...types import Lang


class TTSProvider(ABC):
    """Synthesize `text` in `lang` to an audio file at `out_path` and return it.

    `voice_id` is an optional per-call override. Providers that have no notion
    of a swappable voice (eg macOS `say`) ignore it.
    """

    name: str

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
    ) -> Path: ...

    async def healthcheck(self) -> bool:
        return True
