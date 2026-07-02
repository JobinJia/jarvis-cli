"""Streaming pipeline wiring: `_worker` routing + `_process_one_streaming`.

This path (config flag `behavior.streaming_pipeline`) overlaps LLM token
generation with per-sentence TTS playback. All streamed sentences feed ONE
audio sink (a `StreamPlayer`, or the in-process PCM sink for raw-PCM
providers — spawned lazily) so consecutive sentences play gaplessly; these
tests lock the routing decision, the single-session feed loop, cancel
handling, emotion threading, the per-sentence file fallback, and the
PCM-vs-ffplay sink selection.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon
from jarvis_cli.types import Event


def _llm_event(sid: str = "s1", ntype: str = "permission_prompt") -> Event:
    # text=None → an LLM-phrased event, the only kind the streaming pipeline
    # handles (pre-baked text and session_start keep the batch path).
    return Event(
        notification_type=ntype,
        tool_name="Bash",
        cwd="/repo",
        session_id=sid,
    )


class _FakePrimary:
    """Streaming-capable TTS stub recording each stream() call's text+emotion."""

    name = "fake-tts"
    supports_streaming = True
    stream_input_args = None
    stream_pcm = None  # container-format stream → ffplay path by default

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def stream(
        self, text: str, lang, voice_id=None, emotion=None,
    ) -> AsyncIterator[bytes]:
        self.calls.append((text, emotion))

        async def _gen() -> AsyncIterator[bytes]:
            yield text.encode()

        return _gen()


class _FakeSession:
    """Stand-in for a spawned StreamPlayer, recording feed/close/abort."""

    def __init__(self, feed_error: Exception | None = None) -> None:
        self.fed: list[bytes] = []
        self.closed = False
        self.aborted = False
        self._feed_error = feed_error

    async def feed(self, chunks: AsyncIterator[bytes]) -> None:
        async for chunk in chunks:
            self.fed.append(chunk)
        if self._feed_error is not None:
            err, self._feed_error = self._feed_error, None
            raise err

    async def close(self) -> None:
        self.closed = True

    async def abort(self) -> None:
        self.aborted = True


def _fake_stream_player(first_feed_error: Exception | None = None):
    """Build a StreamPlayer replacement class + the session list it fills.

    `first_feed_error` arms only the FIRST spawned session's first feed to
    raise, so tests can exercise the abort→fallback→fresh-session path.
    """
    sessions: list[_FakeSession] = []

    class _FakeStreamPlayer:
        @classmethod
        async def spawn(cls, *, input_args=None, on_spawn=None) -> _FakeSession:
            err = first_feed_error if not sessions else None
            s = _FakeSession(feed_error=err)
            sessions.append(s)
            if on_spawn is not None:
                on_spawn(MagicMock())
            return s

    return _FakeStreamPlayer, sessions


@pytest.mark.asyncio
async def test_process_one_streaming_feeds_sentences_into_one_session():
    """Both sentences must flow through a SINGLE spawned ffplay session (one
    spawn, one close) — spawning per sentence is what caused audible gaps."""
    d = Daemon(Config())

    async def _phrase_stream(event, *, lang, emotion=None) -> AsyncIterator[str]:
        yield "First sentence here."
        yield "Second sentence here."

    d.router.phrase_stream = _phrase_stream
    primary = _FakePrimary()
    d.tts.primary = primary

    fake_sp, sessions = _fake_stream_player()
    with patch("jarvis_cli.daemon.main.StreamPlayer", fake_sp):
        await d._process_one_streaming(_llm_event())

    assert [text for text, _ in primary.calls] == [
        "First sentence here.", "Second sentence here.",
    ]
    assert len(sessions) == 1, "expected exactly one ffplay session per utterance"
    assert sessions[0].fed == [b"First sentence here.", b"Second sentence here."]
    assert sessions[0].closed is True
    assert sessions[0].aborted is False
    # _last_text is the joined spoken line, used by the webhook + dedup paths.
    assert d._last_text == "First sentence here. Second sentence here."


def test_wants_streaming_predicate():
    """The worker's routing predicate: streaming only for LLM-phrased events
    behind the config flag — pre-baked text and session_start stay batch."""
    cfg = Config()
    cfg.behavior.streaming_pipeline = True
    d = Daemon(cfg)

    assert d._wants_streaming(_llm_event()) is True

    baked = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        cwd="/repo",
        session_id="s1",
        text="pre-baked line",
    )
    assert d._wants_streaming(baked) is False

    start = Event(
        notification_type="session_start",
        tool_name=None,
        cwd="/repo",
        session_id="s1",
    )
    assert d._wants_streaming(start) is False

    d.cfg.behavior.streaming_pipeline = False
    assert d._wants_streaming(_llm_event()) is False


@pytest.mark.asyncio
async def test_streaming_pipeline_flag_routes_to_streaming_path():
    """Drive the real `_worker` loop for one event and assert it dispatches to
    the streaming path — routing lives in production code, not in the test."""
    cfg = Config()
    cfg.behavior.streaming_pipeline = True
    d = Daemon(cfg)

    routed = asyncio.Event()

    async def _streaming(event):
        routed.set()

    with patch.object(d, "_process_one_streaming", side_effect=_streaming), \
            patch.object(d, "_process_one") as batch:
        worker = asyncio.create_task(d._worker())
        await d.queue.put_or_drop(_llm_event())
        await asyncio.wait_for(routed.wait(), timeout=2)
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    batch.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_cancel_mid_stream_stops_playback():
    """A cancel landing between sentences must halt the loop, not keep speaking
    stale sentences for a session the user already moved past — and the shared
    session must still be concluded (closed), never orphaned."""
    d = Daemon(Config())

    async def _phrase_stream(event, *, lang, emotion=None) -> AsyncIterator[str]:
        yield "First sentence here."
        # Simulate the user acting before the second sentence plays.
        d._cancelled_sessions.add("s1")
        yield "Second sentence here."

    d.router.phrase_stream = _phrase_stream
    primary = _FakePrimary()
    d.tts.primary = primary

    fake_sp, sessions = _fake_stream_player()
    with patch("jarvis_cli.daemon.main.StreamPlayer", fake_sp):
        await d._process_one_streaming(_llm_event("s1"))

    assert [text for text, _ in primary.calls] == ["First sentence here."]
    assert len(sessions) == 1
    assert sessions[0].fed == [b"First sentence here."]
    # No orphan ffplay: the session was concluded one way or the other.
    assert sessions[0].closed or sessions[0].aborted


@pytest.mark.asyncio
async def test_streaming_threads_emotion_into_phrase_and_tts():
    """The batch path derives emotion_for(type) and threads it into prompt and
    TTS; the streaming path must do the same — tool_failure maps to 'grave'."""
    d = Daemon(Config())

    seen_phrase_emotions: list[str | None] = []

    async def _phrase_stream(event, *, lang, emotion=None) -> AsyncIterator[str]:
        seen_phrase_emotions.append(emotion)
        yield "A grave development, sir."

    d.router.phrase_stream = _phrase_stream
    primary = _FakePrimary()
    d.tts.primary = primary

    fake_sp, _sessions = _fake_stream_player()
    with patch("jarvis_cli.daemon.main.StreamPlayer", fake_sp):
        await d._process_one_streaming(_llm_event(ntype="tool_failure"))

    assert seen_phrase_emotions == ["grave"]
    assert primary.calls == [("A grave development, sir.", "grave")]


@pytest.mark.asyncio
async def test_streaming_feed_failure_falls_back_to_file_synth():
    """A genuine feed failure (no cancel pending) must abort the shared session
    and fall back to file synth for THAT sentence — with emotion threaded — and
    subsequent sentences retry streaming on a FRESH session."""
    d = Daemon(Config())

    async def _phrase_stream(event, *, lang, emotion=None) -> AsyncIterator[str]:
        yield "First sentence here."
        yield "Second sentence here."

    d.router.phrase_stream = _phrase_stream
    primary = _FakePrimary()
    d.tts.primary = primary

    synth_calls: list[tuple[str, str | None]] = []

    async def _synth(text, lang, out_path, voice_id=None, emotion=None):
        synth_calls.append((text, emotion))
        out_path.write_bytes(b"")  # pretend a wav was produced

    d.tts.synthesize = _synth

    async def _fake_play(audio, *, on_spawn=None):
        return None

    fake_sp, sessions = _fake_stream_player(
        first_feed_error=RuntimeError("pipe went sideways"),
    )
    with patch("jarvis_cli.daemon.main.StreamPlayer", fake_sp), \
            patch("jarvis_cli.daemon.main.play", side_effect=_fake_play):
        await d._process_one_streaming(_llm_event())

    # Sentence 1: fed to session 1, which blew up → aborted, then file synth.
    assert sessions[0].aborted is True
    assert sessions[0].closed is False
    assert synth_calls == [("First sentence here.", "neutral")]
    # Sentence 2: streaming retried on a fresh session, which is closed cleanly.
    assert len(sessions) == 2
    assert sessions[1].fed == [b"Second sentence here."]
    assert sessions[1].closed is True
    assert sessions[1].aborted is False


def _fake_open_pcm_sink(sink: _FakeSession, seen: dict):
    """Stand-in for player.open_pcm_sink recording the PCM spec it was given."""

    async def _open(*, rate, channels, input_args=None, on_spawn=None):
        seen.update(rate=rate, channels=channels, input_args=input_args)
        if on_spawn is not None:
            on_spawn(MagicMock())
        return sink

    return _open


@pytest.mark.asyncio
async def test_process_one_streaming_pcm_provider_routes_through_pcm_sink():
    """A raw-PCM primary (stream_pcm set) must get the in-process sounddevice
    sink — with the provider's rate/channels — instead of an ffplay spawn."""
    d = Daemon(Config())

    async def _phrase_stream(event, *, lang, emotion=None) -> AsyncIterator[str]:
        yield "One sentence, straight to CoreAudio."

    d.router.phrase_stream = _phrase_stream
    primary = _FakePrimary()
    primary.stream_pcm = (24000, 1)
    d.tts.primary = primary

    sink = _FakeSession()
    seen: dict = {}
    with patch(
        "jarvis_cli.daemon.main.open_pcm_sink",
        side_effect=_fake_open_pcm_sink(sink, seen),
    ), patch("jarvis_cli.daemon.main.StreamPlayer") as stream_player:
        await d._process_one_streaming(_llm_event())

    stream_player.spawn.assert_not_called()
    assert seen["rate"] == 24000 and seen["channels"] == 1
    assert sink.fed == [b"One sentence, straight to CoreAudio."]
    assert sink.closed is True


@pytest.mark.asyncio
async def test_try_stream_pcm_provider_routes_through_pcm_sink():
    """The batch whole-text path shares the sink selection: stream_pcm set →
    open_pcm_sink (fed and closed), never play_stream."""
    d = Daemon(Config())
    primary = _FakePrimary()
    primary.stream_pcm = (24000, 1)
    d.tts.primary = primary

    sink = _FakeSession()
    seen: dict = {}
    with patch(
        "jarvis_cli.daemon.main.open_pcm_sink",
        side_effect=_fake_open_pcm_sink(sink, seen),
    ), patch("jarvis_cli.daemon.main.play_stream") as play_stream:
        ok = await d._try_stream("Hello sir.", "en", None)

    assert ok is True
    play_stream.assert_not_called()
    assert seen["rate"] == 24000 and seen["channels"] == 1
    assert sink.fed == [b"Hello sir."]
    assert sink.closed is True
