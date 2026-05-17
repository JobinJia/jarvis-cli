"""XTTS-v2 zero-shot voice-clone TTS via coqui-tts.

Heavy imports (`torch`, `TTS`) are inside `_load_model` to keep the daemon
import-light when XTTS isn't actually used (eg. user opts into ElevenLabs).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from ...config import XTTSConfig
from ...types import Lang
from .base import TTSProvider

_LANG_CODE = {"zh": "zh-cn", "en": "en"}


class XTTSProvider(TTSProvider):
    name = "xtts"

    def __init__(self, cfg: XTTSConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None

    def _ref_audio_for(self, lang: Lang) -> Path:
        path = self.cfg.ref_audio_zh if lang == "zh" else self.cfg.ref_audio_en
        return Path(path)

    def _load_model(self) -> Any:
        """Lazy-load the XTTS-v2 model. Called at most once per provider lifetime."""
        if self._model is not None:
            return self._model
        logger.info("Loading XTTS-v2 model from {} on {}", self.cfg.model_dir, self.cfg.device)
        from TTS.api import TTS  # type: ignore

        self._model = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
        ).to(self.cfg.device)
        return self._model

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
    ) -> Path:
        # XTTS has no notion of an EL-style voice_id (clones from ref audio);
        # the override is ignored. Future: could resolve voice_id to a named
        # ref wav under voices/<voice_id>.wav for multi-voice support.
        _ = voice_id
        ref = self._ref_audio_for(lang)
        if not ref.is_file():
            raise FileNotFoundError(f"reference audio missing: {ref}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        speed = (
            self.cfg.speed_short
            if len(text) < self.cfg.short_threshold_chars
            else self.cfg.speed_long
        )

        def _run() -> None:
            model = self._load_model()
            model.tts_to_file(
                text=text,
                speaker_wav=str(ref),
                language=_LANG_CODE.get(lang, "en"),
                file_path=str(out_path),
                temperature=self.cfg.temperature,
                speed=speed,
            )

        await asyncio.to_thread(_run)
        return out_path

    async def healthcheck(self) -> bool:
        return self._ref_audio_for("zh").is_file() and self._ref_audio_for("en").is_file()
