"""CosyVoice 3 provider unit tests. cosyvoice3 module is not imported here;
the provider does it lazily inside _load_model, and we mock that out so
the tests don't need the (large) Apple Silicon wheel installed in CI."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis_cc.config import CosyVoiceConfig
from jarvis_cc.tts.providers.cosyvoice import CosyVoiceProvider


@pytest.mark.asyncio
async def test_cosyvoice_writes_wav_via_cross_lingual(tmp_path: Path):
    ref = tmp_path / "ref_en.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = CosyVoiceConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref),
        ref_audio_en=str(ref),
        n_timesteps=10,
    )
    p = CosyVoiceProvider(cfg)

    fake_model = MagicMock()
    fake_audio = [0.5, -0.5, 0.0, 1.0, -1.0, 0.25]
    fake_model.inference_cross_lingual = MagicMock(return_value=fake_audio)

    with patch.object(p, "_load_model", return_value=fake_model):
        out = tmp_path / "out.wav"
        result = await p.synthesize("Sir, ready.", lang="en", out_path=out)

    assert result == out
    fake_model.inference_cross_lingual.assert_called_once()
    kwargs = fake_model.inference_cross_lingual.call_args.kwargs
    assert kwargs["text"] == "Sir, ready."
    assert kwargs["prompt_wav"] == str(ref)
    assert kwargs["n_timesteps"] == 10

    # WAV file is mono int16 @ 24kHz
    import wave
    with wave.open(str(out), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 24000
        assert w.getnframes() == len(fake_audio)


@pytest.mark.asyncio
async def test_cosyvoice_raises_if_ref_audio_missing(tmp_path: Path):
    cfg = CosyVoiceConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "missing.wav"),
        ref_audio_en=str(tmp_path / "missing.wav"),
    )
    p = CosyVoiceProvider(cfg)
    with pytest.raises(FileNotFoundError):
        await p.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")


@pytest.mark.asyncio
async def test_cosyvoice_picks_zh_ref_for_chinese_lang(tmp_path: Path):
    ref_en = tmp_path / "ref_en.wav"
    ref_zh = tmp_path / "ref_zh.wav"
    ref_en.write_bytes(b"\x00")
    ref_zh.write_bytes(b"\x00")
    cfg = CosyVoiceConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref_zh),
        ref_audio_en=str(ref_en),
    )
    p = CosyVoiceProvider(cfg)
    fake_model = MagicMock()
    fake_model.inference_cross_lingual = MagicMock(return_value=[0.0])
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize("ni hao", lang="zh", out_path=tmp_path / "o.wav")
    kwargs = fake_model.inference_cross_lingual.call_args.kwargs
    assert kwargs["prompt_wav"] == str(ref_zh)


def test_cosyvoice_advertises_no_streaming():
    """Until we implement streaming, the daemon must NOT try _try_stream
    on this provider."""
    assert CosyVoiceProvider.supports_streaming is False


@pytest.mark.asyncio
async def test_cosyvoice_uses_zero_shot_when_ref_text_provided(tmp_path: Path):
    """A ref transcript routes through inference_zero_shot, which grounds
    the LLM and prevents the double-take loop cross_lingual exhibits on
    short utterances."""
    ref = tmp_path / "ref_en.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = CosyVoiceConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref), ref_audio_en=str(ref),
        ref_text_en="I am Jarvis, a virtual artificial intelligence.",
    )
    p = CosyVoiceProvider(cfg)
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(return_value=[0.0, 0.0])
    fake_model.inference_cross_lingual = MagicMock(return_value=[0.0, 0.0])
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize("Sir, ready.", lang="en", out_path=tmp_path / "o.wav")
    fake_model.inference_zero_shot.assert_called_once()
    fake_model.inference_cross_lingual.assert_not_called()
    kwargs = fake_model.inference_zero_shot.call_args.kwargs
    assert kwargs["prompt_text"] == "I am Jarvis, a virtual artificial intelligence."
    assert kwargs["prompt_wav"] == str(ref)
