"""Synthesis-in-a-child-process: framing, the worker's dispatch, and the
daemon-side handle that replaces a worn child.

Why any of this exists: XTTS leaks ~40 MB of native memory per utterance, and
native memory only returns when a process ends. Recycling the daemon would take
the socket, the event queue and the warmed index with it, so the model was
moved into a disposable child instead. These tests pin the contract that makes
that swap invisible to the playback path.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis_cli.config import Config
from jarvis_cli.tts.protocol import read_frame, write_frame
from jarvis_cli.tts.worker import _Worker
from jarvis_cli.tts.worker_client import WorkerProvider


class _FakeWriter:
    """Records what a frame writer emitted; `drain` is the awaited no-op the
    real StreamWriter provides."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def write(self, data: bytes) -> None:
        self.buf += data

    async def drain(self) -> None:
        return None


def _reader_over(data: bytes) -> asyncio.StreamReader:
    r = asyncio.StreamReader()
    r.feed_data(data)
    r.feed_eof()
    return r


async def _drain_frames(data: bytes) -> list[tuple[dict, bytes]]:
    reader = _reader_over(data)
    out = []
    while (frame := await read_frame(reader)) is not None:
        out.append(frame)
    return out


# ---- framing -------------------------------------------------------------


@pytest.mark.asyncio
async def test_frame_roundtrip_carries_binary_audio_intact():
    """Audio rides as raw bytes after the header, not inside the JSON — a
    newline or an invalid UTF-8 sequence in PCM must not corrupt the stream."""
    w = _FakeWriter()
    payload = bytes(range(256)) * 8 + b"\n{\"not\": \"a header\"}\n"
    await write_frame(w, {"id": 7, "type": "chunk"}, payload)
    await write_frame(w, {"id": 7, "type": "done"})

    frames = await _drain_frames(bytes(w.buf))
    assert [h["type"] for h, _ in frames] == ["chunk", "done"]
    assert frames[0][1] == payload
    # size is derived from the payload, never trusted from the caller
    assert frames[0][0]["size"] == len(payload)


@pytest.mark.asyncio
async def test_read_frame_returns_none_at_eof_and_on_truncation():
    """A child that dies mid-frame must read as EOF, not as a short chunk —
    handing back half an utterance would play as clipped speech."""
    assert await read_frame(_reader_over(b"")) is None

    w = _FakeWriter()
    await write_frame(w, {"id": 1, "type": "chunk"}, b"0123456789")
    truncated = bytes(w.buf)[:-4]
    assert await read_frame(_reader_over(truncated)) is None


# ---- the worker's dispatch ----------------------------------------------


class _FakeProvider:
    name = "fake"

    def __init__(self, chunks=(b"aa", b"bb"), fail: Exception | None = None):
        self._chunks = chunks
        self._fail = fail
        self.prewarmed = False
        self.stopped = False

    async def prewarm(self) -> None:
        self.prewarmed = True

    async def stream(self, text, lang, voice_id=None, emotion=None):
        try:
            for c in self._chunks:
                if self._fail is not None:
                    raise self._fail
                yield c
                await asyncio.sleep(0)
        finally:
            self.stopped = True

    async def synthesize(self, text, lang, out_path, voice_id=None, emotion=None):
        Path(out_path).write_bytes(b"wav")
        return out_path


async def _run_worker(provider, requests: list[dict]) -> list[tuple[dict, bytes]]:
    w = _FakeWriter()
    worker = _Worker(provider, w)
    for req in requests:
        await worker.dispatch(req)
        if worker._current is not None:
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await worker._current
    return await _drain_frames(bytes(w.buf))


@pytest.mark.asyncio
async def test_worker_streams_chunks_then_done():
    p = _FakeProvider()
    frames = await _run_worker(p, [{"id": 1, "op": "stream", "text": "hi", "lang": "en"}])
    assert [h["type"] for h, _ in frames] == ["chunk", "chunk", "done"]
    assert [b for _, b in frames[:2]] == [b"aa", b"bb"]


@pytest.mark.asyncio
async def test_worker_reports_provider_failure_as_an_error_frame():
    """A synthesis failure must come back as data, not as a dead child: the
    daemon's fallback chain can only run if it hears about it."""
    p = _FakeProvider(fail=RuntimeError("model exploded"))
    frames = await _run_worker(p, [{"id": 2, "op": "stream", "text": "x", "lang": "en"}])
    assert frames[-1][0]["type"] == "error"
    assert "model exploded" in frames[-1][0]["message"]


@pytest.mark.asyncio
async def test_worker_prewarm_and_synthesize(tmp_path: Path):
    p = _FakeProvider()
    out = tmp_path / "o.wav"
    frames = await _run_worker(p, [
        {"id": 1, "op": "prewarm"},
        {"id": 2, "op": "synthesize", "text": "x", "lang": "en",
         "out_path": str(out)},
    ])
    assert p.prewarmed is True
    assert [h["type"] for h, _ in frames] == ["done", "done"]
    assert frames[1][0]["path"] == str(out)
    assert out.read_bytes() == b"wav"


@pytest.mark.asyncio
async def test_worker_abort_stops_the_decoder():
    """Cancelling the in-flight task must close the provider's generator —
    that `finally` is what sets the stop flag the real decoder polls between
    pieces, so an abandoned line stops costing GPU time."""
    started = asyncio.Event()

    class _Slow(_FakeProvider):
        async def stream(self, text, lang, voice_id=None, emotion=None):
            try:
                started.set()
                await asyncio.sleep(30)
                yield b"never"
            finally:
                self.stopped = True

    p = _Slow()
    w = _FakeWriter()
    worker = _Worker(p, w)
    await worker.dispatch({"id": 5, "op": "stream", "text": "x", "lang": "en"})
    await asyncio.wait_for(started.wait(), 2)
    await worker.dispatch({"id": 5, "op": "abort"})
    await asyncio.wait_for(worker._current, 2)

    assert p.stopped is True
    frames = await _drain_frames(bytes(w.buf))
    assert frames[-1][0]["type"] == "aborted"


# ---- the daemon-side handle ---------------------------------------------


def _client_with_child(frames: bytes) -> WorkerProvider:
    """A WorkerProvider whose 'child' is a canned frame stream."""
    wp = WorkerProvider("xtts", "/tmp/cfg.toml", max_syntheses=2)
    wp._reader = _reader_over(frames)
    return wp


@pytest.mark.asyncio
async def test_client_yields_chunks_and_counts_the_synthesis():
    w = _FakeWriter()
    await write_frame(w, {"id": 1, "type": "chunk"}, b"pcm-a")
    await write_frame(w, {"id": 1, "type": "chunk"}, b"pcm-b")
    await write_frame(w, {"id": 1, "type": "done"})
    wp = _client_with_child(bytes(w.buf))

    with patch.object(wp, "_send", return_value=None):
        got = [c async for c in wp.stream("hello", "en")]

    assert got == [b"pcm-a", b"pcm-b"]
    assert wp.syntheses == 1


@pytest.mark.asyncio
async def test_client_raises_when_the_child_dies_mid_utterance():
    """EOF on the pipe is a dead child. Raising is what lets the daemon fall
    back to another provider for this line instead of going silent."""
    w = _FakeWriter()
    await write_frame(w, {"id": 1, "type": "chunk"}, b"pcm")
    wp = _client_with_child(bytes(w.buf))

    with patch.object(wp, "_send", return_value=None):
        with pytest.raises(RuntimeError, match="died"):
            _ = [c async for c in wp.stream("hello", "en")]


@pytest.mark.asyncio
async def test_client_aborts_the_child_when_playback_is_cancelled():
    """Closing the generator early is how a cancel reaches us. The child has
    to be told, or it keeps decoding a line nobody will hear."""
    w = _FakeWriter()
    await write_frame(w, {"id": 1, "type": "chunk"}, b"pcm-a")
    await write_frame(w, {"id": 1, "type": "chunk"}, b"pcm-b")
    await write_frame(w, {"id": 1, "type": "done"})
    wp = _client_with_child(bytes(w.buf))

    sent: list[dict] = []

    async def _send(req):
        sent.append(req)

    with patch.object(wp, "_send", side_effect=_send):
        gen = wp.stream("hello", "en")
        assert await gen.__anext__() == b"pcm-a"
        await gen.aclose()

    assert sent[-1] == {"id": 1, "op": "abort"}


def test_recycle_threshold():
    wp = WorkerProvider("xtts", "/tmp/cfg.toml", max_syntheses=2)
    assert wp.should_recycle is False
    wp._syntheses = 2
    assert wp.should_recycle is True
    # 0 means "never recycle" — an explicit opt-out, not an immediate trip.
    forever = WorkerProvider("xtts", "/tmp/cfg.toml", max_syntheses=0)
    forever._syntheses = 10_000
    assert forever.should_recycle is False


def test_client_mirrors_the_provider_streaming_contract():
    """The daemon picks its audio sink from these before any child exists, so
    they must come from the provider class, not from a live worker."""
    from jarvis_cli.tts.providers.xtts import XTTSProvider

    wp = WorkerProvider("xtts", "/tmp/cfg.toml", max_syntheses=100)
    assert wp.supports_streaming is XTTSProvider.supports_streaming
    assert wp.stream_pcm == XTTSProvider.stream_pcm
    assert wp.stream_input_args == XTTSProvider.stream_input_args


# ---- end to end, through a real child process ---------------------------


@pytest.mark.asyncio
async def test_real_child_process_synthesizes_and_recycles(tmp_path: Path):
    """The whole transport for real: spawn a child, get audio back over the
    pipe, then replace it. Uses `say -o` (writes a file, plays nothing) so the
    test needs no model and makes no sound."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    wp = WorkerProvider("say", cfg_path, max_syntheses=1)
    out = tmp_path / "spoken.wav"
    try:
        await asyncio.wait_for(
            wp.synthesize("Worker transport check.", "en", out), 60,
        )
        # Not `wave.open`: `say` writes 32-bit float PCM (format 3), which the
        # stdlib reader rejects. RIFF header + real payload is the point here —
        # that audio crossed the pipe at all.
        head = out.read_bytes()
        assert head[:4] == b"RIFF" and len(head) > 10_000
        assert wp.syntheses == 1
        assert wp.should_recycle is True

        proc = wp._proc
        assert proc is not None and proc.returncode is None
        await wp.recycle()
        assert proc.returncode is not None, "child survived recycle"
        assert wp._proc is None and wp.syntheses == 0
    finally:
        await wp.aclose()


@pytest.mark.asyncio
async def test_real_child_reports_provider_errors_without_dying(tmp_path: Path):
    """A failed line must come back as an error the daemon can fall back on,
    with the child still alive for the next one — restarting on every bad
    utterance would spend 30s of model load to no purpose."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    wp = WorkerProvider("say", cfg_path, max_syntheses=100)
    try:
        with pytest.raises(RuntimeError):
            # An unwritable destination fails inside the provider, not in the
            # transport.
            await asyncio.wait_for(
                wp.synthesize("x", "en", Path("/proc/nope/out.wav")), 60,
            )
        assert wp._proc is not None and wp._proc.returncode is None

        out = tmp_path / "after.wav"
        await asyncio.wait_for(wp.synthesize("Still here.", "en", out), 60)
        assert out.is_file()
    finally:
        await wp.aclose()


# ---- daemon wiring -------------------------------------------------------


def test_heavy_providers_are_wrapped_light_ones_are_not():
    from jarvis_cli.daemon.main import _make_tts_provider

    cfg = Config()
    cfg.tts.worker_process = True
    assert isinstance(_make_tts_provider("xtts", cfg), WorkerProvider)
    assert isinstance(_make_tts_provider("cosyvoice", cfg), WorkerProvider)
    # `say` is a subprocess call already; isolating it would buy nothing.
    assert not isinstance(_make_tts_provider("say", cfg), WorkerProvider)


def test_worker_process_can_be_turned_off():
    from jarvis_cli.daemon.main import _make_tts_provider
    from jarvis_cli.tts.providers.xtts import XTTSProvider

    cfg = Config()
    cfg.tts.worker_process = False
    assert isinstance(_make_tts_provider("xtts", cfg), XTTSProvider)


@pytest.mark.asyncio
async def test_engine_recycles_only_worn_workers():
    from jarvis_cli.tts.engine import TTSEngine

    class _Stub(WorkerProvider):
        def __init__(self, worn: bool) -> None:
            super().__init__("xtts", "/tmp/c.toml", max_syntheses=1)
            self._syntheses = 1 if worn else 0
            self.recycled = False
            self.warmed = False

        async def recycle(self) -> None:
            self.recycled = True

        async def prewarm(self) -> None:
            self.warmed = True

    worn, fresh = _Stub(worn=True), _Stub(worn=False)
    engine = TTSEngine(primary=worn, fallback=None, overrides={"zh": fresh})
    await engine.recycle_worn_workers()

    assert (worn.recycled, worn.warmed) == (True, True)
    assert (fresh.recycled, fresh.warmed) == (False, False)


@pytest.mark.asyncio
async def test_engine_recycle_failure_does_not_break_the_speech_loop():
    from jarvis_cli.tts.engine import TTSEngine

    class _Stuck(WorkerProvider):
        def __init__(self) -> None:
            super().__init__("xtts", "/tmp/c.toml", max_syntheses=1)
            self._syntheses = 5

        async def recycle(self) -> None:
            raise RuntimeError("child will not die")

    engine = TTSEngine(primary=_Stuck(), fallback=None)
    await engine.recycle_worn_workers()  # must not raise
