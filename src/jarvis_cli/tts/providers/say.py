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

# `say` picks the container from the file extension. .aiff is its native
# format and works with no flags. .wav needs an explicit data format —
# without it, say errors out with "Opening output file failed: fmt?".
_DATA_FORMAT_BY_SUFFIX = {
    ".wav": "LEF32@22050",  # 32-bit little-endian float PCM, 22.05kHz
}


class SayProvider(TTSProvider):
    name = "say"

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> Path:
        # `voice_id` here is interpreted as a macOS `say` voice name (eg "Daniel",
        # "Tingting", "Karen"); fall back to a language-appropriate default.
        voice = voice_id or _VOICE_BY_LANG.get(lang, "Daniel")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        argv: list[str] = ["say", "-v", voice, "-o", str(out_path)]
        data_format = _DATA_FORMAT_BY_SUFFIX.get(out_path.suffix.lower())
        if data_format is not None:
            argv.append(f"--data-format={data_format}")
        argv.append(text)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"say failed: {err.decode(errors='replace')}")
        return out_path
