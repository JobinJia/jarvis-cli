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

import numpy as np
from loguru import logger

from ...config import CosyVoiceConfig
from ...types import Lang
from ..duration_guard import _MIN_CLEAN_CPS, DurationBaseline, DurationVerdict
from .base import TTSProvider

_SAMPLE_RATE = 24000  # CosyVoice 3 fixed output rate (mono float).

# CosyVoice's flow decoder occasionally "double-takes" on a synth — it emits
# the line, or just its tail, twice in one clip (~12% of short lines even via
# inference_zero_shot), which makes the audio run much longer than normal. We
# detect it by duration: each synth is compared against the text's clean
# baseline (see tts/duration_guard.py) and resynthesized if it runs past
# baseline x duration_ratio_threshold. (An SSM self-similarity approach was
# tried and discarded — on real audio it couldn't tell a double-take apart
# from the ordinary self-similarity of normal long speech.)


class CosyVoiceProvider(TTSProvider):
    name = "cosyvoice"

    def __init__(self, cfg: CosyVoiceConfig) -> None:
        self.cfg = cfg
        self._model: Any | None = None
        self._baseline = DurationBaseline(
            Path(cfg.duration_baseline_path).expanduser()
        )

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
        emotion: str | None = None,
    ) -> Path:
        # voice_id ignored: CosyVoice clones the prompt audio's voice; the
        # English/Chinese ref wavs are the only voice knob the daemon exposes.
        _ = voice_id
        ref = self._ref_audio_for(lang)
        if not ref.is_file():
            raise FileNotFoundError(f"reference audio missing: {ref}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        prompt_text = self._ref_text_for(lang).strip()

        # Synthesize up to max_synth_attempts times; accept as soon as the take
        # isn't a double-take. The verdict is by duration vs the text's clean
        # baseline — all lengths checked. If every attempt is flagged we ship
        # the SHORTEST take (least likely to contain an audible repeat) so the
        # daemon always speaks.
        best_audio: np.ndarray | None = None
        best_duration = 0.0
        for attempt in range(1, self.cfg.max_synth_attempts + 1):
            audio = await asyncio.to_thread(
                self._synth_once, text, prompt_text, ref, out_path,
            )
            duration = len(audio) / _SAMPLE_RATE if len(audio) else 0.0
            verdict = self._baseline.check(
                text, duration,
                ratio_threshold=self.cfg.duration_ratio_threshold,
                fallback_cps=self.cfg.fallback_cps,
            )
            logger.info(
                "CosyVoice synth: chars={} duration={:.2f}s expected={:.2f}s "
                "ratio={:.2f} text={!r}",
                len(text), duration, verdict.expected, verdict.ratio, text[:60],
            )
            if self.cfg.save_synth_samples:
                self._save_sample(
                    text, lang, out_path, duration, verdict, attempt,
                )
            if duration > 0 and (best_audio is None or duration < best_duration):
                best_audio, best_duration = audio, duration
            if not verdict.is_repeat:
                # Clean takes update the rolling-window baseline.
                self._baseline.record(text, duration)
                if attempt > 1:
                    logger.info(
                        "CosyVoice retry succeeded on attempt {} (ratio={:.2f})",
                        attempt, verdict.ratio,
                    )
                return out_path
            logger.warning(
                "CosyVoice double-take suspected (ratio={:.2f}, expected={:.2f}s, "
                "attempt {}/{}); resynthesizing text={!r}",
                verdict.ratio, verdict.expected, attempt,
                self.cfg.max_synth_attempts, text[:60],
            )

        # All attempts flagged: ship the shortest take. Escape valve — learn
        # its duration if its pace is plausibly clean (gated by _MIN_CLEAN_CPS),
        # so a baseline that drifted too low can recover instead of flagging
        # this text forever. A genuine double-take (too slow) is left unlearned.
        if best_audio is not None:
            self._write_wav(out_path, best_audio)
            self._baseline.record(text, best_duration, min_cps=_MIN_CLEAN_CPS)
            logger.warning(
                "CosyVoice gave up after {} attempts; shipping shortest take "
                "({:.2f}s) text={!r}",
                self.cfg.max_synth_attempts, best_duration, text[:60],
            )
        return out_path

    def _synth_once(
        self, text: str, prompt_text: str, ref: Path, out_path: Path,
    ) -> np.ndarray:
        """One model forward pass + WAV write. Returns the audio as a float32
        array so the caller can measure its duration without re-reading the
        file."""
        model = self._load_model()
        # zero_shot when we have a transcript of the ref audio — the LLM uses
        # the transcript to know where the prompt audio ends, which reduces
        # (but does not eliminate — see the retry loop in `synthesize`) the
        # double-take the decoder falls into on short utterances. Without a
        # transcript we fall back to cross_lingual.
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
        # `audio` is a sequence of floats in [-1, 1]. Convert to a float32 array
        # and write a standard mono WAV.
        arr = np.asarray(audio, dtype=np.float32).clip(-1.0, 1.0)
        self._write_wav(out_path, arr)
        return arr

    @staticmethod
    def _write_wav(out_path: Path, arr: np.ndarray) -> None:
        """Write a float32 [-1, 1] array as a mono int16 WAV at _SAMPLE_RATE."""
        pcm = (arr * 32767.0).astype("<i2").tobytes()
        with wave.open(str(out_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            w.writeframes(pcm)

    def _save_sample(
        self,
        text: str,
        lang: Lang,
        src_wav: Path,
        duration: float,
        verdict: DurationVerdict,
        attempt: int,
    ) -> None:
        """Dump a synth + metadata sidecar to `sample_dir` for offline
        analysis. Best-effort: a sampling failure must never break synthesis."""
        import hashlib
        import json
        import shutil
        import time

        try:
            out_dir = Path(self.cfg.sample_dir).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = (
                f"{time.time():.3f}_"
                f"{hashlib.sha1(text.encode()).hexdigest()[:8]}_a{attempt}"
            )
            shutil.copyfile(src_wav, out_dir / f"{stamp}.wav")
            (out_dir / f"{stamp}.json").write_text(
                json.dumps(
                    {
                        "text": text,
                        "lang": lang,
                        "attempt": attempt,
                        "duration_s": duration,
                        "expected_s": verdict.expected,
                        "ratio": verdict.ratio,
                        "is_repeat": verdict.is_repeat,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except OSError as e:
            logger.warning("CosyVoice sample save failed: {}", e)

    async def healthcheck(self) -> bool:
        return (
            self._ref_audio_for("zh").is_file()
            and self._ref_audio_for("en").is_file()
        )
