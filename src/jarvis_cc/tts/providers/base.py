"""Abstract base for TTS providers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ...types import Lang


class TTSProvider(ABC):
    """Synthesize `text` in `lang` to an audio file at `out_path` and return it."""

    name: str

    @abstractmethod
    async def synthesize(self, text: str, lang: Lang, out_path: Path) -> Path: ...

    async def healthcheck(self) -> bool:
        return True
