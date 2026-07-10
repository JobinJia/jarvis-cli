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
async def test_xtts_synthesize_emotion_shapes_prosody(tmp_path: Path):
    """The batch/embedding path applies the same emotion → prosody mapping as
    streaming: "grave" (tool_failure) slows delivery and cuts sampling
    variance so bad news is read straight."""
    import numpy as np

    pth = tmp_path / "jarvis_speaker.pth"
    pth.write_bytes(b"\x00")
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

    with patch.object(p, "_load_model", return_value=fake_tts), patch.object(
        p, "_load_latents", return_value=latents
    ):
        await p.synthesize(
            "The build failed, sir.", lang="en",
            out_path=tmp_path / "out.wav", emotion="grave",
        )

    kwargs = fake_tts.synthesizer.tts_model.inference.call_args.kwargs
    # grave → (×0.92, -0.05) on top of the short-text base speed.
    assert kwargs["speed"] == pytest.approx(cfg.speed_short * 0.92)
    assert kwargs["temperature"] == pytest.approx(
        min(max(cfg.temperature - 0.05, 0.3), 0.85)
    )


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


@pytest.mark.asyncio
async def test_xtts_stream_yields_pcm_chunks(tmp_path: Path):
    """Streaming path advertises supports_streaming and emits 16-bit PCM
    bytes, one per chunk the GPT decoder produces, decode hints in tow."""
    import numpy as np

    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "z.wav"),
        ref_audio_en=str(tmp_path / "e.wav"),
        speaker_embedding="",
        device="cpu",
    )
    p = XTTSProvider(cfg)
    assert p.supports_streaming is True
    # Probe-skipping flags must precede the format spec — they tell ffplay
    # to trust the explicit s16le description and start decoding immediately
    # instead of buffering input to sniff the format. Mono MUST be spelled
    # -ch_layout (ffplay has no -ac; passing it makes ffplay exit at spawn).
    assert p.stream_input_args == (
        "-probesize", "32",
        "-analyzeduration", "0",
        "-fflags", "nobuffer",
        "-f", "s16le", "-ar", "24000", "-ch_layout", "mono",
    )
    # Same byte stream described for the in-process sounddevice sink.
    assert p.stream_pcm == (24000, 1)

    class _Chunk:
        def __init__(self, arr): self._arr = arr
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self._arr

    fake_model = MagicMock()
    fake_model.synthesizer.tts_model.inference_stream.return_value = iter(
        [_Chunk(np.array([0.0, 1.0, -1.0], dtype=np.float32))]
    )

    with patch.object(p, "_load_model", return_value=fake_model), \
            patch.object(p, "_conditioning_for", return_value=("g", "s")):
        chunks = [c async for c in p.stream("hello", lang="en")]

    assert len(chunks) == 2
    # First item is the amp-wake pre-roll: 0.25s of silent int16 mono PCM,
    # so the speaker wake-up swallows padding rather than the first phoneme.
    assert chunks[0] == bytes(2 * int(24000 * 0.25))
    # 3 float samples → 3 int16 → 6 bytes; 1.0 → 32767, -1.0 → -32767.
    assert chunks[1] == np.array([0, 32767, -32767], dtype=np.int16).tobytes()
    kwargs = fake_model.synthesizer.tts_model.inference_stream.call_args.kwargs
    assert kwargs["language"] == "en"
    assert kwargs["gpt_cond_latent"] == "g"
    assert kwargs["speaker_embedding"] == "s"
    # Config value passes through verbatim (20 on MPS — see XTTSConfig for
    # the measured chunk-size trade-off).
    assert kwargs["stream_chunk_size"] == 20
    # No emotion → prosody untouched: base short-text speed, base temperature.
    assert kwargs["speed"] == pytest.approx(cfg.speed_short)
    assert kwargs["temperature"] == pytest.approx(cfg.temperature)


@pytest.mark.asyncio
async def test_xtts_stream_emotion_shapes_prosody(tmp_path: Path):
    """Emotion must reach the streaming decoder as prosody nudges: "pleased"
    (the brightest tone in the vocabulary) multiplies speed and lifts the
    sampling temperature, both within the safe clamp window."""
    import numpy as np

    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "z.wav"),
        ref_audio_en=str(tmp_path / "e.wav"),
        speaker_embedding="",
        device="cpu",
    )
    p = XTTSProvider(cfg)

    class _Chunk:
        def __init__(self, arr): self._arr = arr
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self._arr

    fake_model = MagicMock()
    fake_model.synthesizer.tts_model.inference_stream.return_value = iter(
        [_Chunk(np.zeros(3, dtype=np.float32))]
    )

    with patch.object(p, "_load_model", return_value=fake_model), \
            patch.object(p, "_conditioning_for", return_value=("g", "s")):
        _ = [c async for c in p.stream("All done, sir.", lang="en", emotion="pleased")]

    kwargs = fake_model.synthesizer.tts_model.inference_stream.call_args.kwargs
    # pleased → (×1.05, +0.08) on top of the short-text base speed.
    assert kwargs["speed"] == pytest.approx(cfg.speed_short * 1.05)
    assert kwargs["temperature"] == pytest.approx(
        min(max(cfg.temperature + 0.08, 0.3), 0.85)
    )


@pytest.mark.asyncio
async def test_xtts_stream_propagates_inference_error(tmp_path: Path):
    """An exception inside the worker-thread generator surfaces to the async
    consumer rather than hanging — the daemon relies on this to fall back."""
    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "z.wav"),
        ref_audio_en=str(tmp_path / "e.wav"),
        speaker_embedding="",
        device="cpu",
    )
    p = XTTSProvider(cfg)
    fake_model = MagicMock()
    fake_model.synthesizer.tts_model.inference_stream.side_effect = RuntimeError("boom")

    with patch.object(p, "_load_model", return_value=fake_model), \
            patch.object(p, "_conditioning_for", return_value=("g", "s")):
        with pytest.raises(RuntimeError, match="boom"):
            async for _ in p.stream("hello", lang="en"):
                pass


def test_split_for_gpt_passes_short_text_through():
    from jarvis_cli.tts.providers.xtts import _split_for_gpt

    assert _split_for_gpt("Right away, sir.") == ["Right away, sir."]


def test_split_for_gpt_splits_long_text_at_sentence_boundaries():
    """XTTS's GPT silently truncates past ~250 chars — long announcements
    lost their tail until we started splitting. Every piece must fit the
    limit and no text may be dropped."""
    from jarvis_cli.tts.providers.xtts import _GPT_CHAR_LIMIT, _split_for_gpt

    long = " ".join(
        f"Sentence number {i} reporting in with a reasonable length, sir."
        for i in range(12)
    )
    assert len(long) > _GPT_CHAR_LIMIT
    pieces = _split_for_gpt(long)
    assert len(pieces) >= 2
    assert all(len(pc) <= _GPT_CHAR_LIMIT for pc in pieces)
    # Nothing dropped: rejoined pieces reproduce the input (whitespace-joined).
    assert " ".join(pieces) == long


def test_split_for_gpt_hard_splits_single_overlong_sentence():
    from jarvis_cli.tts.providers.xtts import _GPT_CHAR_LIMIT, _split_for_gpt

    monster = "word " * 100  # no sentence-end punctuation at all
    pieces = _split_for_gpt(monster)
    assert all(len(pc) <= _GPT_CHAR_LIMIT for pc in pieces)
    assert "".join(pieces).replace(" ", "") == monster.strip().replace(" ", "")


@pytest.mark.asyncio
async def test_xtts_stream_generates_per_piece_for_long_text(tmp_path: Path):
    """A long text must drive one inference_stream call per split piece —
    feeding it whole is exactly the silent-truncation bug."""
    import numpy as np

    cfg = XTTSConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(tmp_path / "z.wav"),
        ref_audio_en=str(tmp_path / "e.wav"),
        speaker_embedding="",
        device="cpu",
    )
    p = XTTSProvider(cfg)

    class _Chunk:
        def __init__(self, arr): self._arr = arr
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self._arr

    long_text = " ".join(
        f"Sentence number {i} reporting in with a reasonable length, sir."
        for i in range(12)
    )
    fake_model = MagicMock()
    fake_model.synthesizer.tts_model.inference_stream.side_effect = lambda **kw: iter(
        [_Chunk(np.array([0.5], dtype=np.float32))]
    )

    with patch.object(p, "_load_model", return_value=fake_model), \
            patch.object(p, "_conditioning_for", return_value=("g", "s")):
        chunks = [c async for c in p.stream(long_text, lang="en")]

    from jarvis_cli.tts.providers.xtts import _split_for_gpt
    n_pieces = len(_split_for_gpt(long_text))
    assert fake_model.synthesizer.tts_model.inference_stream.call_count == n_pieces
    texts = [
        c.kwargs["text"]
        for c in fake_model.synthesizer.tts_model.inference_stream.call_args_list
    ]
    assert " ".join(texts) == long_text
    # pre-roll + one PCM chunk per piece
    assert len(chunks) == 1 + n_pieces


# --- dash-pause normalization -------------------------------------------------
# Measured 2026-07-09 (3 runs per case): XTTS renders an em-dash as a dead
# stop of 0.42-1.2s (variance included) vs 0.36-0.56s for a comma. Dashes in
# the written line become commas in the text handed to the GPT.


def test_normalize_pauses_en_dash_to_comma():
    from jarvis_cli.tts.providers.xtts import _normalize_pauses

    assert _normalize_pauses(
        "Sir, the build failed — three tests did not pass.", "en",
    ) == "Sir, the build failed, three tests did not pass."


def test_normalize_pauses_zh_double_dash_to_comma():
    from jarvis_cli.tts.providers.xtts import _normalize_pauses

    assert _normalize_pauses(
        "先生，测试未通过——三个用例失败了。", "zh",
    ) == "先生，测试未通过，三个用例失败了。"


def test_normalize_pauses_keeps_hyphens_and_plain_text():
    from jarvis_cli.tts.providers.xtts import _normalize_pauses

    line = "Sir, he wants to run pre-commit with --no-verify."
    assert _normalize_pauses(line, "en") == line


def test_normalize_pauses_trailing_dash_does_not_dangle():
    from jarvis_cli.tts.providers.xtts import _normalize_pauses

    assert _normalize_pauses(
        "Sir, he wishes to push to main —", "en",
    ) == "Sir, he wishes to push to main"


def test_stream_receives_normalized_text():
    """The GPT must never see the dash — verify at the inference boundary."""
    import numpy as np

    class _Chunk:
        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    p = XTTSProvider(XTTSConfig(speaker_embedding="dummy.pth"))
    fake_model = MagicMock()
    fake_model.synthesizer.tts_model.inference_stream.side_effect = lambda **kw: iter(
        [_Chunk(np.array([0.5], dtype=np.float32))]
    )

    async def _run():
        with patch.object(p, "_load_model", return_value=fake_model), \
                patch.object(p, "_conditioning_for", return_value=("g", "s")):
            return [c async for c in p.stream("A — B.", lang="en")]

    import asyncio
    asyncio.run(_run())
    sent = fake_model.synthesizer.tts_model.inference_stream.call_args.kwargs["text"]
    assert sent == "A, B."
