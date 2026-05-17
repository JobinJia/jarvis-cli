from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis_cc.config import XTTSConfig
from jarvis_cc.tts.providers.xtts import XTTSProvider


@pytest.mark.asyncio
async def test_xtts_calls_underlying_engine(tmp_path: Path):
    ref = tmp_path / "ref_zh.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref),
        ref_audio_en=str(ref),
        device="cpu",
    )
    p = XTTSProvider(cfg)

    fake_tts = MagicMock()
    fake_tts.tts_to_file = MagicMock(return_value=None)

    with patch.object(p, "_load_model", return_value=fake_tts):
        out = tmp_path / "out.wav"
        result = await p.synthesize("hello", lang="zh", out_path=out)

    assert result == out
    fake_tts.tts_to_file.assert_called_once()
    kwargs = fake_tts.tts_to_file.call_args.kwargs
    assert kwargs["text"] == "hello"
    assert kwargs["language"] == "zh-cn"
    assert kwargs["speaker_wav"] == str(ref)
    assert kwargs["file_path"] == str(out)
    # Temperature must be forwarded from config to inference — XTTS's
    # library default 0.75 produces noticeably more pacing/intonation
    # variance than we want.
    assert kwargs["temperature"] == pytest.approx(0.5)
    # "hello" is 5 chars → falls into the short bucket, so speed_short
    # (1.30) is what reaches the engine, not speed_long.
    assert kwargs["speed"] == pytest.approx(1.30)


@pytest.mark.asyncio
async def test_xtts_picks_speed_long_for_long_text(tmp_path: Path):
    """Long text gets a different (lower) speed multiplier because XTTS's
    GPT already speeds long utterances up on its own — applying speed_short
    to them turns long readouts into auctioneer-pace.
    """
    ref = tmp_path / "ref_en.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref), ref_audio_en=str(ref),
        device="cpu",
    )
    long_text = "Sir, " + "this is a deliberately long status update. " * 5
    assert len(long_text) >= cfg.short_threshold_chars

    p = XTTSProvider(cfg)
    fake_tts = MagicMock()
    fake_tts.tts_to_file = MagicMock(return_value=None)
    with patch.object(p, "_load_model", return_value=fake_tts):
        await p.synthesize(long_text, lang="en", out_path=tmp_path / "o.wav")

    kwargs = fake_tts.tts_to_file.call_args.kwargs
    assert kwargs["speed"] == pytest.approx(cfg.speed_long)


@pytest.mark.asyncio
async def test_xtts_raises_if_ref_audio_missing(tmp_path: Path):
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "missing.wav"),
        ref_audio_en=str(tmp_path / "missing.wav"),
        device="cpu",
    )
    p = XTTSProvider(cfg)
    with pytest.raises(FileNotFoundError):
        await p.synthesize("hi", lang="zh", out_path=tmp_path / "o.wav")
