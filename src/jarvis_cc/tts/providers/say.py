"""macOS `say` provider: lowest-quality but always-available fallback."""
from __future__ import annotations

import asyncio
from pathlib import Path

from ...types import Lang
from .base import TTSProvider

_VOICE_BY_LANG = {
    "en": "Daniel",   # British male
    "zh": "Tingting",  # Mandarin female
}


class SayProvider(TTSProvider):
    name = "say"

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
    ) -> Path:
        # `voice_id` here is interpreted as a macOS `say` voice name (eg "Daniel",
        # "Tingting", "Karen"); fall back to a language-appropriate default.
        voice = voice_id or _VOICE_BY_LANG.get(lang, "Daniel")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # `say` outputs AIFF, but afplay handles both .aiff and .wav.
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", voice, "-o", str(out_path), text,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"say failed: {err.decode(errors='replace')}")
        return out_path
