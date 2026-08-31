"""Piper TTS provider unit tests. The `piper` package is not imported here;
the provider loads it lazily inside `_load_voice`, and we mock that out so the
tests don't need the ONNX voice models installed in CI.

Piper synthesizes from a FIXED single-speaker model, so unlike the CosyVoice
zero-shot clone it cannot drift in accent and never double-takes - there is
no retry/duration guard to test, just a straight model-forward to WAV.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis.config import PiperConfig
from jarvis.tts.providers.piper import PiperProvider


def _fake_voice(frames: int = 24000) -> MagicMock:
    """A stand-in PiperVoice whose synthesize_wav writes a mono int16 clip."""
    voice = MagicMock()

    def _synth(text, wav_file, *a, **k):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * frames)

    voice.synthesize_wav = MagicMock(side_effect=_synth)
    return voice


@pytest.mark.asyncio
async def test_piper_writes_wav_via_voice(tmp_path: Path):
    cfg = PiperConfig(data_dir=str(tmp_path), voice_en="en_GB-alan-medium")
    p = PiperProvider(cfg)
    voice = _fake_voice()
    with patch.object(p, "_load_voice", return_value=voice) as load:
        out = tmp_path / "out.wav"
        result = await p.synthesize("Sir, ready.", lang="en", out_path=out)

    assert result == out
    load.assert_called_once_with("en_GB-alan-medium")
    voice.synthesize_wav.assert_called_once()
    assert voice.synthesize_wav.call_args.args[0] == "Sir, ready."
    import wave
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 24000


@pytest.mark.asyncio
async def test_piper_picks_zh_voice_for_chinese(tmp_path: Path):
    cfg = PiperConfig(
        data_dir=str(tmp_path),
        voice_en="en_GB-alan-medium",
        voice_zh="zh_CN-huayan-medium",
    )
    p = PiperProvider(cfg)
    with patch.object(p, "_load_voice", return_value=_fake_voice()) as load:
        await p.synthesize("ni hao", lang="zh", out_path=tmp_path / "o.wav")
    load.assert_called_once_with("zh_CN-huayan-medium")


@pytest.mark.asyncio
async def test_piper_caches_loaded_voice(tmp_path: Path):
    """The ONNX model loads at most once per voice across calls - the daemon
    keeps it resident, which is what makes per-utterance RTF ~0.03."""
    cfg = PiperConfig(data_dir=str(tmp_path), voice_en="en_GB-alan-medium")
    p = PiperProvider(cfg)
    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"\x00")  # presence check
    voice = _fake_voice()
    with patch(
        "jarvis.tts.providers.piper.PiperVoice"
    ) as PV:
        PV.load = MagicMock(return_value=voice)
        await p.synthesize("one", lang="en", out_path=tmp_path / "a.wav")
        await p.synthesize("two", lang="en", out_path=tmp_path / "b.wav")
    PV.load.assert_called_once()  # loaded once, reused on the second call


@pytest.mark.asyncio
async def test_piper_prewarm_loads_and_caches_en_voice(tmp_path: Path):
    """prewarm() eagerly loads the English voice at daemon start, so the first
    notification reuses the resident model instead of paying the ONNX load."""
    cfg = PiperConfig(data_dir=str(tmp_path), voice_en="en_GB-alan-medium")
    p = PiperProvider(cfg)
    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"\x00")  # presence check
    voice = _fake_voice()
    with patch(
        "jarvis.tts.providers.piper.PiperVoice"
    ) as PV:
        PV.load = MagicMock(return_value=voice)
        await p.prewarm()
        PV.load.assert_called_once()  # en voice loaded eagerly
        await p.synthesize("Sir, ready.", lang="en", out_path=tmp_path / "a.wav")
    PV.load.assert_called_once()  # synth reused the prewarmed cache


def test_piper_advertises_no_streaming():
    assert PiperProvider.supports_streaming is False


@pytest.mark.asyncio
async def test_piper_healthcheck_requires_voice_file(tmp_path: Path):
    cfg = PiperConfig(data_dir=str(tmp_path), voice_en="en_GB-alan-medium")
    p = PiperProvider(cfg)
    assert await p.healthcheck() is False  # no .onnx on disk yet
    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"\x00")
    assert await p.healthcheck() is True
