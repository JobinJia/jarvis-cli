"""Piper TTS (rhasspy) fixed-speaker provider via the MIT `piper-tts` wheel.

Piper runs an ONNX voice on CPU (no PyTorch, no Metal) and renders from a
single-speaker model, so — unlike the CosyVoice zero-shot clone — the accent is
baked into the weights and cannot drift, and there is no flow-decoder
double-take to retry. The model stays resident after the first load, giving a
warm per-utterance RTF of ~0.03 (≈30x realtime).

Voices live as `<name>.onnx` (+ `<name>.onnx.json`) under `cfg.data_dir`;
fetch one with `python -m piper.download_voices <name> --data-dir <data_dir>`.
"""
from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from typing import Any

from loguru import logger

from ...config import PiperConfig
from ...types import Lang
from .base import TTSProvider

# Imported lazily-by-name so unit tests can patch it without the wheel present.
try:  # pragma: no cover - exercised indirectly; absence is the fallback path
    from piper import PiperVoice
except Exception:  # noqa: BLE001 - any import failure means "not installed yet"
    PiperVoice = None  # type: ignore[assignment]


class PiperProvider(TTSProvider):
    name = "piper"

    def __init__(self, cfg: PiperConfig) -> None:
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir).expanduser()
        # One resident PiperVoice per voice name (en/zh share this cache).
        self._voices: dict[str, Any] = {}

    def _voice_name(self, lang: Lang) -> str:
        return self.cfg.voice_zh if lang == "zh" else self.cfg.voice_en

    def _voice_path(self, name: str) -> Path:
        return self.data_dir / f"{name}.onnx"

    def _load_voice(self, name: str) -> Any:
        """Load and cache the ONNX voice. Called at most once per voice name —
        the model is kept resident so synthesis stays fast."""
        cached = self._voices.get(name)
        if cached is not None:
            return cached
        if PiperVoice is None:
            raise RuntimeError(
                "piper-tts is not installed; `uv pip install piper-tts`"
            )
        path = self._voice_path(name)
        if not path.is_file():
            raise FileNotFoundError(
                f"Piper voice missing: {path}. Fetch it with "
                f"`python -m piper.download_voices {name} --data-dir {self.data_dir}`"
            )
        logger.info("Loading Piper voice {} from {}", name, path)
        voice = PiperVoice.load(str(path))
        self._voices[name] = voice
        return voice

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
    ) -> Path:
        # voice_id overrides the per-lang voice name when given (eg a request
        # for a specific downloaded voice); otherwise pick by language.
        name = voice_id or self._voice_name(lang)
        voice = await asyncio.to_thread(self._load_voice, name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _synth() -> None:
            with wave.open(str(out_path), "wb") as w:
                voice.synthesize_wav(text, w)

        await asyncio.to_thread(_synth)
        return out_path

    async def healthcheck(self) -> bool:
        # Healthy when the English voice is on disk (zh is optional). We don't
        # load the model here — file presence is enough to know synthesis can
        # start, and loading is deferred to the first synth.
        return self._voice_path(self.cfg.voice_en).is_file()
