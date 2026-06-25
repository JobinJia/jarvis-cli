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
        self._latents: dict[str, Any] | None = None

    def _ref_audio_for(self, lang: Lang) -> Path:
        path = self.cfg.ref_audio_zh if lang == "zh" else self.cfg.ref_audio_en
        return Path(path)

    def _embedding_path(self, lang: Lang) -> Path | None:
        """Configured speaker-embedding `.pth` for `lang`, or None.

        The bundled Jarvis (Bettany) embedding only sounds right in English —
        speaking Chinese through it is muddy — so Chinese always falls back to
        the ref-audio clone path regardless of `speaker_embedding`.
        """
        if lang == "zh" or not self.cfg.speaker_embedding:
            return None
        path = Path(self.cfg.speaker_embedding)
        return path if path.is_file() else None

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

    def _load_latents(self) -> dict[str, Any]:
        """Lazy-load the pre-extracted speaker embedding onto the model device."""
        if self._latents is not None:
            return self._latents
        import torch  # type: ignore

        path = self.cfg.speaker_embedding
        logger.info("Loading XTTS speaker embedding from {}", path)
        # The .pth is a plain dict of tensors (not weights-only safe), hence
        # weights_only=False. Move both latents onto the synthesis device so
        # inference doesn't trip over a CPU/MPS tensor mismatch.
        raw = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self._latents = {
            "gpt_cond_latent": raw["gpt_cond_latent"].to(self.cfg.device),
            "speaker_embedding": raw["speaker_embedding"].to(self.cfg.device),
        }
        return self._latents

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> Path:
        # XTTS has no notion of an EL-style voice_id (clones from ref audio /
        # embedding); the override is ignored. Future: could resolve voice_id
        # to a named embedding under voices/<voice_id>.pth for multi-voice.
        _ = voice_id
        out_path.parent.mkdir(parents=True, exist_ok=True)

        speed = (
            self.cfg.speed_short
            if len(text) < self.cfg.short_threshold_chars
            else self.cfg.speed_long
        )
        language = _LANG_CODE.get(lang, "en")
        embedding = self._embedding_path(lang)

        if embedding is None:
            ref = self._ref_audio_for(lang)
            if not ref.is_file():
                raise FileNotFoundError(f"reference audio missing: {ref}")

            def _run_ref() -> None:
                model = self._load_model()
                model.tts_to_file(
                    text=text,
                    speaker_wav=str(ref),
                    language=language,
                    file_path=str(out_path),
                    temperature=self.cfg.temperature,
                    speed=speed,
                )

            await asyncio.to_thread(_run_ref)
            return out_path

        def _run_embedding() -> None:
            import numpy as np  # type: ignore
            from scipy.io import wavfile  # type: ignore

            tts = self._load_model()
            latents = self._load_latents()
            out = tts.synthesizer.tts_model.inference(
                text=text,
                language=language,
                gpt_cond_latent=latents["gpt_cond_latent"],
                speaker_embedding=latents["speaker_embedding"],
                temperature=self.cfg.temperature,
                speed=speed,
            )
            wav = np.asarray(out["wav"], dtype=np.float32)
            # Clip rather than peak-normalise: peak-normalising each utterance
            # independently makes loudness jump between lines. XTTS output is
            # already ~[-1, 1]; occasional overshoot is clamped here.
            wav = np.clip(wav, -1.0, 1.0)
            sr = int(getattr(tts.synthesizer, "output_sample_rate", 24000))
            wavfile.write(str(out_path), sr, (wav * 32767).astype(np.int16))

        await asyncio.to_thread(_run_embedding)
        return out_path

    async def healthcheck(self) -> bool:
        # English is served by the embedding when present; zh always needs its
        # ref wav. Stay healthy as long as English can be produced.
        en_ok = self._embedding_path("en") is not None or self._ref_audio_for("en").is_file()
        return en_ok
