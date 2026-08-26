"""CosyVoice 3 provider unit tests. cosyvoice3 module is not imported here;
the provider does it lazily inside _load_model, and we mock that out so
the tests don't need the (large) Apple Silicon wheel installed in CI."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis_cli.config import CosyVoiceConfig
from jarvis_cli.tts.providers.cosyvoice import CosyVoiceProvider


@pytest.mark.asyncio
async def test_cosyvoice_writes_wav_via_cross_lingual(tmp_path: Path):
    ref = tmp_path / "ref_en.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = CosyVoiceConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref),
        ref_audio_en=str(ref),
        n_timesteps=10,
        duration_baseline_path=str(tmp_path / "baseline.json"),
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
        duration_baseline_path=str(tmp_path / "baseline.json"),
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
        duration_baseline_path=str(tmp_path / "baseline.json"),
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


# ---------------------------------------------------------------------------
# Double-take detection (the bug behind the user-reported "repeated playback").
# Detection is by DURATION (see tts/duration_guard.py): a take running well
# past the text's clean baseline is a repeat. So these tests drive the model
# with silence of a given length — duration is all the detector reads.
# ---------------------------------------------------------------------------


def _silence(seconds: float) -> list[float]:
    """A flat clip of a given length; duration is the only signal that matters."""
    return [0.0] * int(seconds * 24000)


def _doubletake_cfg(tmp_path: Path) -> CosyVoiceConfig:
    ref = tmp_path / "ref_en.wav"
    ref.write_bytes(b"\x00" * 1024)
    return CosyVoiceConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref), ref_audio_en=str(ref),
        ref_text_en="(ref transcript)",
        duration_ratio_threshold=1.35,
        fallback_cps=12.0,
        max_synth_attempts=4,
        # Isolate each test: empty baseline file under tmp, not the real cache.
        duration_baseline_path=str(tmp_path / "baseline.json"),
        sample_dir=str(tmp_path / "samples"),
    )


@pytest.mark.asyncio
async def test_cosyvoice_retries_when_too_long(tmp_path: Path):
    """A take running well past the text's expected duration is a double-take
    (the model said the line twice) and must be retried."""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    # 33-char text → char-fallback expected ≈ 33/12 = 2.75s, threshold ≈ 3.71s.
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(side_effect=[
        _silence(4.0),  # too long → repeat
        _silence(2.0),  # normal → clean
    ])
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(
            "Sir, Claude awaits your guidance.",
            lang="en", out_path=tmp_path / "out.wav",
        )

    assert fake_model.inference_zero_shot.call_count == 2
    import wave
    with wave.open(str(tmp_path / "out.wav"), "rb") as w:
        assert w.getnframes() == len(_silence(2.0))  # the clean second take


@pytest.mark.asyncio
async def test_cosyvoice_caps_retry_attempts(tmp_path: Path):
    """If every take is too long, stop at max_synth_attempts and ship the last —
    a heard double-take beats a hung daemon."""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(return_value=_silence(4.0))
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(
            "Sir, Claude awaits your guidance.",
            lang="en", out_path=tmp_path / "out.wav",
        )
    assert fake_model.inference_zero_shot.call_count == cfg.max_synth_attempts


@pytest.mark.asyncio
async def test_cosyvoice_checks_long_inputs_too(tmp_path: Path):
    """Long lines are checked too — there is no length cap. (The old cps
    heuristic skipped >80-char lines, leaving long-line double-takes
    unguarded — e.g. the logged 147-char case.)"""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    long_text = ("This is a much longer briefing line that goes on for "
                 "rather more than the eighty character validation cap.")
    assert len(long_text) > 80
    # ~106 chars → expected ≈ 8.8s, threshold ≈ 11.9s.
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(side_effect=[
        _silence(14.0),  # too long → repeat
        _silence(8.0),   # normal → clean
    ])
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(long_text, lang="en", out_path=tmp_path / "out.wav")
    assert fake_model.inference_zero_shot.call_count == 2


@pytest.mark.asyncio
async def test_cosyvoice_first_call_accepted_when_normal_length(tmp_path: Path):
    """No retry when the first take is a normal length — common-case latency
    unchanged."""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(return_value=_silence(2.0))
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(
            "Sir, Claude awaits your guidance.",
            lang="en", out_path=tmp_path / "out.wav",
        )
    assert fake_model.inference_zero_shot.call_count == 1


@pytest.mark.asyncio
async def test_cosyvoice_baseline_catches_what_char_fallback_misses(
    tmp_path: Path,
):
    """A learned per-text baseline is tighter than the char estimate: a take
    that char-fallback alone would pass gets caught once the real clean length
    is known."""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    text = "Sir, Claude awaits your guidance."  # char expected 2.75s, thr 3.71s

    # Teach the baseline a 2.2s clean take (char-fallback passes it too).
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(return_value=_silence(2.2))
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(text, lang="en", out_path=tmp_path / "a.wav")

    # A 3.0s take: char-fallback (thr 3.71) would PASS, but the baseline
    # (2.2 * 1.35 = 2.97) catches it → retry.
    fake_model.inference_zero_shot = MagicMock(side_effect=[
        _silence(3.0), _silence(2.2),
    ])
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(text, lang="en", out_path=tmp_path / "b.wav")
    assert fake_model.inference_zero_shot.call_count == 2


@pytest.mark.asyncio
async def test_cosyvoice_saves_samples_when_enabled(tmp_path: Path):
    """With save_synth_samples on, each take's wav + a metadata sidecar
    (text, duration, expected, ratio, verdict) is dumped to sample_dir."""
    import json

    cfg = _doubletake_cfg(tmp_path)
    cfg.save_synth_samples = True
    p = CosyVoiceProvider(cfg)
    fake_model = MagicMock()
    # 11-char text, 1.0s → char threshold ≈ 1.24s, so this is clean (1 call).
    fake_model.inference_zero_shot = MagicMock(return_value=_silence(1.0))
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize("Sir, ready.", lang="en", out_path=tmp_path / "o.wav")

    wavs = list((tmp_path / "samples").glob("*.wav"))
    metas = list((tmp_path / "samples").glob("*.json"))
    assert len(wavs) == 1
    assert len(metas) == 1
    d = json.loads(metas[0].read_text())
    assert d["text"] == "Sir, ready."
    assert d["lang"] == "en"
    assert "duration_s" in d
    assert "ratio" in d
    assert d["is_repeat"] is False


@pytest.mark.asyncio
async def test_cosyvoice_no_samples_when_disabled(tmp_path: Path):
    """Sampling is off by default — no files written, no overhead."""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(return_value=_silence(1.0))
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize("Sir, ready.", lang="en", out_path=tmp_path / "o.wav")
    assert not (tmp_path / "samples").exists()


@pytest.mark.asyncio
async def test_cosyvoice_ships_shortest_take_when_all_flagged(tmp_path: Path):
    """When every attempt is flagged a repeat, ship the SHORTEST take — it is
    the least likely to contain an audible double-take — not whichever take
    happened to be last."""
    import wave

    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    text = "Sir, Claude awaits your guidance."  # char est 2.75s, thr 3.71s
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(side_effect=[
        _silence(5.0), _silence(4.0), _silence(6.0), _silence(4.5),
    ])
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(text, lang="en", out_path=tmp_path / "out.wav")
    assert fake_model.inference_zero_shot.call_count == cfg.max_synth_attempts
    with wave.open(str(tmp_path / "out.wav"), "rb") as w:
        assert w.getnframes() == len(_silence(4.0))  # the shortest, not the last


@pytest.mark.asyncio
async def test_cosyvoice_escape_valve_unsticks_too_low_baseline(tmp_path: Path):
    """A baseline set too low (one fast fluke) flags every normal take forever,
    because a clean take is never recorded to correct it. When retries exhaust,
    the shortest plausibly-clean take is recorded — raising the median so the
    text un-sticks on the next synth."""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    text = "Sir, Claude awaits your guidance."  # 33 chars
    # Seed a too-low baseline: a single fast fluke at 1.6s (median = 1.6,
    # threshold 1.6*1.35 = 2.16s) makes normal ~2.5s takes look like repeats.
    p._baseline.record(text, 1.6)

    # Round 1: every take is a normal-paced 2.5s (33/2.5 = 13.2 cps) but still
    # over 2.16s → all flagged, retries exhaust, escape valve learns 2.5s.
    fake1 = MagicMock()
    fake1.inference_zero_shot = MagicMock(return_value=_silence(2.5))
    with patch.object(p, "_load_model", return_value=fake1):
        await p.synthesize(text, lang="en", out_path=tmp_path / "a.wav")
    assert fake1.inference_zero_shot.call_count == cfg.max_synth_attempts

    # Round 2: the median has risen (1.6 and 2.5 → 2.05s), so a 2.5s take
    # (ratio 1.22) is now accepted on the first try. The trap is broken.
    fake2 = MagicMock()
    fake2.inference_zero_shot = MagicMock(return_value=_silence(2.5))
    with patch.object(p, "_load_model", return_value=fake2):
        await p.synthesize(text, lang="en", out_path=tmp_path / "b.wav")
    assert fake2.inference_zero_shot.call_count == 1


@pytest.mark.asyncio
async def test_cosyvoice_escape_valve_ignores_genuine_double_take(tmp_path: Path):
    """The escape valve must not learn from a take that is itself too slow to
    be clean (a real double-take). Such a take leaves the baseline untouched."""
    cfg = _doubletake_cfg(tmp_path)
    p = CosyVoiceProvider(cfg)
    text = "Sir, Claude awaits your guidance."  # 33 chars
    # Every take is 5.0s → 33/5.0 = 6.6 cps, below the clean floor → a real
    # double-take. Exhaust retries; the baseline must stay empty (char fallback).
    fake_model = MagicMock()
    fake_model.inference_zero_shot = MagicMock(return_value=_silence(5.0))
    with patch.object(p, "_load_model", return_value=fake_model):
        await p.synthesize(text, lang="en", out_path=tmp_path / "out.wav")
    v = p._baseline.check(text, 2.5, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.expected != 5.0  # nothing learned from the double-take
    assert abs(v.expected - 33 / 12.0) < 0.01  # still pure char fallback


@pytest.mark.asyncio
async def test_cosyvoice_loads_the_model_without_prewarm(tmp_path: Path):
    """The daemon defers prewarm for a zh override the config can't route to
    (see Daemon._tts_prewarm_plan); that only holds because synthesize loads
    the model itself. Lock it: no prewarm() call, model still loaded."""
    ref = tmp_path / "ref_en.wav"
    ref.write_bytes(b"\x00" * 1024)
    cfg = CosyVoiceConfig(
        model_dir=str(tmp_path / "model"),
        ref_audio_zh=str(ref),
        ref_audio_en=str(ref),
        duration_baseline_path=str(tmp_path / "baseline.json"),
    )
    p = CosyVoiceProvider(cfg)
    assert p._model is None  # nothing warmed it

    loads = []

    def _fake_load():
        loads.append(1)
        model = MagicMock()
        model.inference_cross_lingual = MagicMock(return_value=[0.1] * 8)
        p._model = model
        return model

    with patch.object(p, "_load_model", _fake_load):
        await p.synthesize("Sir.", lang="zh", out_path=tmp_path / "o.wav")

    assert loads == [1]
