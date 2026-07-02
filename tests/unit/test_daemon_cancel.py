"""Daemon.cancel_session: kill current proc + drop same-sid queued events."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.types import Event


def _ev(sid: str | None, tool: str = "T") -> Event:
    return Event(
        notification_type="permission_prompt",
        tool_name=tool,
        cwd=f"/{sid}",
        session_id=sid,
    )


@pytest.mark.asyncio
async def test_cancel_session_kills_current_proc_for_matching_sid():
    d = Daemon(Config())
    proc = MagicMock()
    proc.kill = MagicMock()
    d._current_proc = proc
    d._current_session_id = "abc"

    await d.cancel_session("abc")

    proc.kill.assert_called_once()
    assert "abc" in d._cancelled_sessions


@pytest.mark.asyncio
async def test_cancel_session_does_not_kill_proc_for_other_sid():
    d = Daemon(Config())
    proc = MagicMock()
    proc.kill = MagicMock()
    d._current_proc = proc
    d._current_session_id = "abc"

    await d.cancel_session("xyz")

    proc.kill.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_session_drops_matching_queued_events():
    d = Daemon(Config())
    await d.queue.put_or_drop(_ev("abc", tool="T1"))
    await d.queue.put_or_drop(_ev("xyz", tool="T2"))
    await d.queue.put_or_drop(_ev("abc", tool="T3"))

    await d.cancel_session("abc")

    survivors = []
    while d.queue.size:
        survivors.append((await d.queue.get()).tool_name)
    assert survivors == ["T2"]


@pytest.mark.asyncio
async def test_try_stream_returns_true_when_session_cancelled():
    """Regression: when the worker kills ffplay (the cancel path), play_stream
    raises and `_try_stream` previously returned False, which sent the worker
    into the synth+afplay fallback — the same line got replayed. Now it must
    return True so the worker skips the fallback."""
    d = Daemon(Config())
    d._cancelled_sessions.add("abc")

    # Stub a streaming primary; play_stream is patched to raise.
    d.tts.primary = MagicMock()
    d.tts.primary.supports_streaming = True
    d.tts.primary.stream_pcm = None  # container stream → ffplay path

    async def _stream_iter(*args, **kwargs):
        yield b"x"

    d.tts.primary.stream = _stream_iter

    from unittest.mock import patch

    async def _fake_play_stream(chunks, *, on_spawn=None):
        # Drain the iterator so it doesn't warn about un-awaited generator.
        async for _ in chunks:
            pass
        raise RuntimeError("ffplay exited with code -9")

    with patch("jarvis_cli.daemon.main.play_stream", side_effect=_fake_play_stream):
        result = await d._try_stream("hi", "en", None, session_id="abc")

    assert result is True


@pytest.mark.asyncio
async def test_try_stream_returns_false_on_real_tts_failure():
    """A genuine TTS error (no cancel pending) must still return False so the
    worker can fall back to synth+afplay."""
    d = Daemon(Config())  # _cancelled_sessions is empty

    d.tts.primary = MagicMock()
    d.tts.primary.supports_streaming = True
    d.tts.primary.stream_pcm = None  # container stream → ffplay path

    async def _stream_iter(*args, **kwargs):
        yield b"x"

    d.tts.primary.stream = _stream_iter

    from unittest.mock import patch

    async def _fake_play_stream(chunks, *, on_spawn=None):
        async for _ in chunks:
            pass
        raise RuntimeError("elevenlabs 500")

    with patch("jarvis_cli.daemon.main.play_stream", side_effect=_fake_play_stream):
        result = await d._try_stream("hi", "en", None, session_id="other-sid")

    assert result is False


@pytest.mark.asyncio
async def test_worker_skips_play_when_cancelled_during_synth():
    """The reported gap: a non-streaming provider (CosyVoice) synthesizes to a
    file — several seconds — with no play proc yet registered. If the user acts
    mid-synth, the cancel kills nothing, and the stale line plays a step late.
    `_process_one` must re-check after synth and drop before `play`."""
    d = Daemon(Config())
    d.tts.primary = MagicMock()
    d.tts.primary.supports_streaming = False  # forces the synth+play fallback

    ev = Event(
        notification_type="permission_prompt",
        tool_name="T", session_id="abc", text="hi", lang="en",
    )

    async def _synth(text, lang, out_path, voice_id=None, emotion=None):
        out_path.write_bytes(b"")  # pretend a file was produced
        d._cancelled_sessions.add("abc")  # user acts mid-synthesis

    d.tts.synthesize = _synth

    from unittest.mock import patch

    play_calls = 0

    async def _fake_play(audio, *, on_spawn=None):
        nonlocal play_calls
        play_calls += 1

    with patch("jarvis_cli.daemon.main.play", side_effect=_fake_play):
        await d._process_one(ev)

    assert play_calls == 0


@pytest.mark.asyncio
async def test_worker_skips_stream_when_cancelled_during_phrasing():
    """A streaming provider's phrasing/LLM step can also outlast the user. If a
    cancel lands during phrasing — before playback begins — `_process_one` must
    not even start the stream."""
    d = Daemon(Config())
    d.tts.primary = MagicMock()
    d.tts.primary.supports_streaming = True

    # Un-prebaked event so the worker takes the phrasing branch we hook into.
    ev = Event(
        notification_type="permission_prompt",
        tool_name="T", session_id="abc",
    )

    async def _phrase(event, *, lang, emotion=None):
        d._cancelled_sessions.add("abc")  # user acts during phrasing
        return "hello"

    d.router.phrase = _phrase

    from unittest.mock import patch

    stream_calls = 0

    async def _guarded_try_stream(*args, **kwargs):
        nonlocal stream_calls
        stream_calls += 1
        return True

    with patch.object(d, "_try_stream", side_effect=_guarded_try_stream):
        await d._process_one(ev)

    assert stream_calls == 0


@pytest.mark.asyncio
async def test_cancel_session_handles_process_lookup_error():
    d = Daemon(Config())

    proc = MagicMock()
    proc.kill = MagicMock(side_effect=ProcessLookupError())
    d._current_proc = proc
    d._current_session_id = "abc"

    # Should not raise
    await d.cancel_session("abc")
