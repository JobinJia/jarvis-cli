"""CosyVoice 3 zero-shot voice-clone TTS via the Apache-2.0 `cosyvoice3` wheel
(Rust+Candle, Apple Silicon Metal). No PyTorch dependency.

A/B testing showed clearer Bettany-clone fidelity than XTTS-v2 at the cost
of ~2-3x synthesis latency. We accept the latency to unlock Apache-2.0 for
the OSS path (XTTS's CPML blocks commercial reuse downstream).
"""
from __future__ import annotations

import asyncio
import wave
from pathlib import Path
from typing import Any

from loguru import logger

from ...config import CosyVoiceConfig
from ...types import Lang
from .base import TTSProvider

_SAMPLE_RATE = 24000  # CosyVoice 3 fixed output rate (mono float).

# CosyVoice's `inference_zero_shot` (used when `ref_text_en` is provided)
# still occasionally double-takes on short inputs — the model emits the
# line twice in one audio. Empirically ~12% on lines under 50 chars even
# at n_timesteps=10. We detect it post-synth by measuring chars-per-second:
# natural English speech is ~12-16 cps; doubled audio falls to ~5-7. We
# accept the audio if cps is above _MIN_ACCEPTABLE_CPS or if it's run out
# of retries. Retries are cheap (one model forward pass) so we permit a
# few; full latency cost is bounded by `_MAX_SYNTH_ATTEMPTS`.
_MIN_ACCEPTABLE_CPS = 9.0
_MAX_SYNTH_ATTEMPTS = 3
# Don't bother validating cps on long lines: the double-take effect is
# rare for inputs longer than this, and the cps signal gets noisy because
# punctuation pauses pull the average down.
_VALIDATION_MAX_CHARS = 80


class CosyVoiceProvider(TTSProvider):
    name = "cosyvoice"

    def __init__(self, cfg: CosyVoiceConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None

    def _ref_audio_for(self, lang: Lang) -> Path:
        path = self.cfg.ref_audio_zh if lang == "zh" else self.cfg.ref_audio_en
        return Path(path)

    def _ref_text_for(self, lang: Lang) -> str:
        return self.cfg.ref_text_zh if lang == "zh" else self.cfg.ref_text_en

    def _load_model(self) -> Any:
        """Lazy-load the CosyVoice 3 model. Called at most once per provider
        lifetime — the underlying wheel auto-detects Metal."""
        if self._model is not None:
            return self._model
        logger.info("Loading CosyVoice3 from {} (Metal)", self.cfg.model_dir)
        import cosyvoice3  # type: ignore

        self._model = cosyvoice3.CosyVoice3(str(Path(self.cfg.model_dir).expanduser()))
        return self._model

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
    ) -> Path:
        # voice_id ignored: CosyVoice clones the prompt audio's voice; the
        # English/Chinese ref wavs are the only voice knob the daemon exposes.
        _ = voice_id
        ref = self._ref_audio_for(lang)
        if not ref.is_file():
            raise FileNotFoundError(f"reference audio missing: {ref}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_text = self._ref_text_for(lang).strip()
        validate = len(text) <= _VALIDATION_MAX_CHARS

        # Run synth up to N times; bail out as soon as the result passes
        # the double-take check (or on the last attempt regardless).
        last_duration = 0.0
        for attempt in range(1, _MAX_SYNTH_ATTEMPTS + 1):
            last_duration = await asyncio.to_thread(
                self._synth_once, text, prompt_text, ref, out_path,
            )
            if not validate:
                break
            cps = (len(text) / last_duration) if last_duration > 0 else 0
            if cps >= _MIN_ACCEPTABLE_CPS:
                if attempt > 1:
                    logger.info(
                        "CosyVoice retry succeeded on attempt {} "
                        "(cps={:.1f}, text={!r})",
                        attempt, cps, text[:60],
                    )
                break
            logger.warning(
                "CosyVoice double-take suspected (cps={:.1f}, attempt {}/{}); "
                "resynthesizing text={!r}",
                cps, attempt, _MAX_SYNTH_ATTEMPTS, text[:60],
            )
        return out_path

    def _synth_once(
        self, text: str, prompt_text: str, ref: Path, out_path: Path,
    ) -> float:
        """One model forward pass + WAV write. Returns audio duration in
        seconds so the caller can validate against text length."""
        model = self._load_model()
        # zero_shot when we have a transcript of the ref audio — the
        # LLM uses the transcript to know where the prompt audio ends,
        # which sharply reduces the "double-take" loop cross_lingual falls
        # into on short utterances. (It does NOT eliminate it — see the
        # retry loop in `synthesize`.) Without a transcript we fall back
        # to cross_lingual and accept the known artifact.
        if prompt_text:
            audio = model.inference_zero_shot(
                text=text,
                prompt_text=prompt_text,
                prompt_wav=str(ref),
                n_timesteps=self.cfg.n_timesteps,
            )
        else:
            audio = model.inference_cross_lingual(
                text=text,
                prompt_wav=str(ref),
                n_timesteps=self.cfg.n_timesteps,
            )
        # `audio` is a Python list of floats in [-1, 1]. Convert to
        # int16 little-endian PCM and write a standard WAV.
        try:
            import numpy as np  # type: ignore
            arr = np.asarray(audio, dtype=np.float32).clip(-1.0, 1.0)
            pcm = (arr * 32767.0).astype("<i2").tobytes()
        except ImportError:
            import struct
            pcm = b"".join(
                struct.pack("<h", max(-32768, min(32767, int(s * 32767))))
                for s in audio
            )
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            w.writeframes(pcm)
        duration_s = len(audio) / _SAMPLE_RATE
        # cps is the post-synth signal the retry loop validates against;
        # logged so we can monitor double-take frequency in the field.
        cps = (len(text) / duration_s) if duration_s > 0 else 0
        logger.info(
            "CosyVoice synth: chars={} duration={:.2f}s cps={:.1f} text={!r}",
            len(text), duration_s, cps, text[:60],
        )
        return duration_s

    async def healthcheck(self) -> bool:
        return (
            self._ref_audio_for("zh").is_file()
            and self._ref_audio_for("en").is_file()
        )
