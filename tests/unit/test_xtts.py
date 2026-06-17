from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis_cli.config import XTTSConfig
from jarvis_cli.tts.providers.xtts import XTTSProvider


@pytest.mark.asyncio
async def test_xtts_calls_underlying_engine(tmp_path: Path):
    ref = tmp_path / "ref_zh.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref),
        ref_audio_en=str(ref),
        speaker_embedding="",
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
    # (1.15) is what reaches the engine, not speed_long.
    assert kwargs["speed"] == pytest.approx(1.15)


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
        speaker_embedding="",
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
        speaker_embedding="",
        device="cpu",
    )
    p = XTTSProvider(cfg)
    with pytest.raises(FileNotFoundError):
        await p.synthesize("hi", lang="zh", out_path=tmp_path / "o.wav")


@pytest.mark.asyncio
async def test_xtts_uses_speaker_embedding_when_present(tmp_path: Path):
    """With a speaker_embedding configured and present, the provider clones
    from the cached latents via inference() and writes the wav itself —
    never touching the ref-audio / tts_to_file path.
    """
    import numpy as np

    pth = tmp_path / "jarvis_speaker.pth"
    pth.write_bytes(b"\x00")  # presence-only; _load_latents is mocked below
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "missing.wav"),
        ref_audio_en=str(tmp_path / "missing.wav"),
        speaker_embedding=str(pth),
        device="cpu",
    )
    p = XTTSProvider(cfg)

    fake_tts = MagicMock()
    fake_tts.synthesizer.output_sample_rate = 24000
    fake_tts.synthesizer.tts_model.inference = MagicMock(
        return_value={"wav": np.zeros(2400, dtype=np.float32)}
    )
    latents = {"gpt_cond_latent": object(), "speaker_embedding": object()}

    out = tmp_path / "out.wav"
    with patch.object(p, "_load_model", return_value=fake_tts), patch.object(
        p, "_load_latents", return_value=latents
    ):
        result = await p.synthesize("hello", lang="en", out_path=out)

    assert result == out
    assert out.is_file()  # provider wrote the wav itself
    fake_tts.tts_to_file.assert_not_called()
    kwargs = fake_tts.synthesizer.tts_model.inference.call_args.kwargs
    assert kwargs["text"] == "hello"
    assert kwargs["language"] == "en"
    assert kwargs["gpt_cond_latent"] is latents["gpt_cond_latent"]
    assert kwargs["speaker_embedding"] is latents["speaker_embedding"]
    assert kwargs["temperature"] == pytest.approx(0.5)
    assert kwargs["speed"] == pytest.approx(1.15)


@pytest.mark.asyncio
async def test_xtts_falls_back_to_ref_when_embedding_missing(tmp_path: Path):
    """A configured-but-absent embedding path must not hijack synthesis —
    the provider falls back to the ref-audio clone path.
    """
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref), ref_audio_en=str(ref),
        speaker_embedding=str(tmp_path / "nope.pth"),
        device="cpu",
    )
    p = XTTSProvider(cfg)
    fake_tts = MagicMock()
    with patch.object(p, "_load_model", return_value=fake_tts):
        await p.synthesize("hi", lang="en", out_path=tmp_path / "o.wav")

    fake_tts.tts_to_file.assert_called_once()
    fake_tts.synthesizer.tts_model.inference.assert_not_called()


@pytest.mark.asyncio
async def test_xtts_chinese_skips_embedding_uses_ref(tmp_path: Path):
    """The Bettany embedding is English-only; Chinese must ignore it and clone
    from ref_audio_zh instead (it sounds muddy speaking Chinese).
    """
    ref = tmp_path / "ref_zh.wav"
    ref.write_bytes(b"\x00" * 1024)
    pth = tmp_path / "jarvis_speaker.pth"
    pth.write_bytes(b"\x00")  # present, but must not be used for zh
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref), ref_audio_en=str(ref),
        speaker_embedding=str(pth),
        device="cpu",
    )
    p = XTTSProvider(cfg)
    fake_tts = MagicMock()
    with patch.object(p, "_load_model", return_value=fake_tts):
        await p.synthesize("你好", lang="zh", out_path=tmp_path / "o.wav")

    fake_tts.tts_to_file.assert_called_once()
    assert fake_tts.tts_to_file.call_args.kwargs["language"] == "zh-cn"
    assert fake_tts.tts_to_file.call_args.kwargs["speaker_wav"] == str(ref)
    fake_tts.synthesizer.tts_model.inference.assert_not_called()
