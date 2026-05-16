"""Daemon.cancel_session: kill current proc + drop same-sid queued events."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis_cc.config import Config
from jarvis_cc.daemon.main import Daemon
from jarvis_cc.types import Event


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

    async def _stream_iter(*args, **kwargs):
        yield b"x"

    d.tts.primary.stream = _stream_iter

    from unittest.mock import patch

    async def _fake_play_stream(chunks, *, on_spawn=None):
        # Drain the iterator so it doesn't warn about un-awaited generator.
        async for _ in chunks:
            pass
        raise RuntimeError("ffplay exited with code -9")

    with patch("jarvis_cc.daemon.main.play_stream", side_effect=_fake_play_stream):
        result = await d._try_stream("hi", "en", None, session_id="abc")

    assert result is True


@pytest.mark.asyncio
async def test_try_stream_returns_false_on_real_tts_failure():
    """A genuine TTS error (no cancel pending) must still return False so the
    worker can fall back to synth+afplay."""
    d = Daemon(Config())  # _cancelled_sessions is empty

    d.tts.primary = MagicMock()
    d.tts.primary.supports_streaming = True

    async def _stream_iter(*args, **kwargs):
        yield b"x"

    d.tts.primary.stream = _stream_iter

    from unittest.mock import patch

    async def _fake_play_stream(chunks, *, on_spawn=None):
        async for _ in chunks:
            pass
        raise RuntimeError("elevenlabs 500")

    with patch("jarvis_cc.daemon.main.play_stream", side_effect=_fake_play_stream):
        result = await d._try_stream("hi", "en", None, session_id="other-sid")

    assert result is False


@pytest.mark.asyncio
async def test_cancel_session_handles_process_lookup_error():
    d = Daemon(Config())

    proc = MagicMock()
    proc.kill = MagicMock(side_effect=ProcessLookupError())
    d._current_proc = proc
    d._current_session_id = "abc"

    # Should not raise
    await d.cancel_session("abc")
