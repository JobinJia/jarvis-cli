import asyncio
import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jarvis_cli.player import play


@pytest.mark.asyncio
async def test_play_invokes_afplay_with_path(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    calls = []

    async def _fake_exec(*args, **kwargs):
        calls.append(args)

        class _P:
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    with patch("jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        await play(audio)

    assert calls[0] == ("afplay", str(audio))


@pytest.mark.asyncio
async def test_play_stream_pipes_chunks_to_ffplay():
    """play_stream(chunks) spawns ffplay reading from stdin and feeds each
    chunk as it arrives — first audio plays before the iterator finishes."""
    from jarvis_cli.player import play_stream

    written: list[bytes] = []
    closed = {"value": False}

    class _Stdin:
        async def drain(self):
            return None

        def write(self, data: bytes):
            written.append(data)

        def close(self):
            closed["value"] = True

        async def wait_closed(self):
            return None

        def is_closing(self):
            return closed["value"]

    spawn_args: list[tuple] = []

    async def _fake_exec(*args, **kwargs):
        spawn_args.append(args)

        class _P:
            returncode = 0
            stdin = _Stdin()

            async def wait(self):
                return 0

        return _P()

    async def _chunks():
        yield b"chunk-A"
        yield b"chunk-B"

    with patch(
        "jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        await play_stream(_chunks())

    # First positional arg is the player binary; flag set marks the streaming
    # invocation (stdin pipe, no display, autoexit).
    assert spawn_args, "ffplay was never spawned"
    assert spawn_args[0][0] == "ffplay"
    assert "-autoexit" in spawn_args[0]
    assert "-nodisp" in spawn_args[0]
    assert b"".join(written) == b"chunk-Achunk-B"
    assert closed["value"] is True


@pytest.mark.asyncio
async def test_play_stream_raises_when_ffplay_fails():
    from jarvis_cli.player import play_stream

    async def _fake_exec(*args, **kwargs):
        class _Stdin:
            async def drain(self):
                return None

            def write(self, data: bytes):
                pass

            def close(self):
                pass

            async def wait_closed(self):
                return None

            def is_closing(self):
                return False

        class _P:
            returncode = 1
            stdin = _Stdin()
            stderr = None  # close() reads it for the error detail when set

            async def wait(self):
                return 1

        return _P()

    async def _chunks():
        yield b"x"

    with patch(
        "jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        with pytest.raises(RuntimeError):
            await play_stream(_chunks())


@pytest.mark.asyncio
async def test_play_invokes_on_spawn_with_proc(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    seen = []

    class _P:
        returncode = 0

        async def communicate(self):
            return (b"", b"")

        async def wait(self):
            return 0

    async def _fake_exec(*args, **kwargs):
        return _P()

    with patch("jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        await play(audio, on_spawn=seen.append)

    assert len(seen) == 1
    assert isinstance(seen[0], _P)


@pytest.mark.asyncio
async def test_play_stream_invokes_on_spawn_with_proc():
    from jarvis_cli.player import play_stream
    seen = []

    class _Stdin:
        async def drain(self): return None
        def write(self, data: bytes): pass
        def close(self): pass
        async def wait_closed(self): return None
        def is_closing(self): return False

    class _P:
        returncode = 0
        stdin = _Stdin()

        async def wait(self): return 0

    async def _fake_exec(*args, **kwargs):
        return _P()

    async def _chunks():
        yield b"x"

    with patch(
        "jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        await play_stream(_chunks(), on_spawn=seen.append)

    assert len(seen) == 1
    assert isinstance(seen[0], _P)


@pytest.mark.asyncio
async def test_stream_player_one_proc_across_multiple_feeds():
    """The whole point of StreamPlayer: several chunk iterators (per-sentence
    TTS streams) flow through ONE ffplay pipe — no per-sentence respawn gap."""
    from jarvis_cli.player import StreamPlayer

    written: list[bytes] = []
    closed = {"value": False}

    class _Stdin:
        async def drain(self):
            return None

        def write(self, data: bytes):
            written.append(data)

        def close(self):
            closed["value"] = True

        async def wait_closed(self):
            return None

        def is_closing(self):
            return closed["value"]

    spawn_count = {"value": 0}

    async def _fake_exec(*args, **kwargs):
        spawn_count["value"] += 1

        class _P:
            returncode = 0
            stdin = _Stdin()

            async def wait(self):
                return 0

        return _P()

    async def _sentence(data: bytes):
        yield data

    with patch(
        "jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        player = await StreamPlayer.spawn()
        await player.feed(_sentence(b"first."))
        await player.feed(_sentence(b"second."))
        await player.close()

    assert spawn_count["value"] == 1
    assert b"".join(written) == b"first.second."
    assert closed["value"] is True


@pytest.mark.asyncio
async def test_stream_player_close_raises_on_nonzero_exit():
    from jarvis_cli.player import StreamPlayer

    class _Stdin:
        async def drain(self): return None
        def write(self, data: bytes): pass
        def close(self): pass
        async def wait_closed(self): return None
        def is_closing(self): return False

    class _P:
        returncode = 1
        stdin = _Stdin()
        stderr = None  # close() reads it for the error detail when set

        async def wait(self): return 1

    async def _fake_exec(*args, **kwargs):
        return _P()

    with patch(
        "jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        player = await StreamPlayer.spawn()
        with pytest.raises(RuntimeError):
            await player.close()


@pytest.mark.asyncio
async def test_stream_player_abort_never_raises():
    """abort() is the error-path cleanup — a dead process (ProcessLookupError
    from kill) must be swallowed, not mask the exception being handled."""
    from jarvis_cli.player import StreamPlayer

    class _Stdin:
        async def drain(self): return None
        def write(self, data: bytes): pass
        def close(self): pass
        async def wait_closed(self): return None
        def is_closing(self): return False

    class _P:
        returncode = -9
        stdin = _Stdin()

        def kill(self):
            raise ProcessLookupError()

        async def wait(self): return -9

    async def _fake_exec(*args, **kwargs):
        return _P()

    with patch(
        "jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        player = await StreamPlayer.spawn()
        await player.abort()  # must not raise


@pytest.mark.asyncio
async def test_stream_player_spawn_passes_input_args_before_pipe():
    """Headerless streams (XTTS raw PCM) need decode flags positioned before
    `-i pipe:0`, or ffplay misparses the byte stream."""
    from jarvis_cli.player import StreamPlayer

    class _Stdin:
        async def drain(self): return None
        def write(self, data: bytes): pass
        def close(self): pass
        async def wait_closed(self): return None
        def is_closing(self): return False

    class _P:
        returncode = 0
        stdin = _Stdin()

        async def wait(self): return 0

    spawn_args: list[tuple] = []

    async def _fake_exec(*args, **kwargs):
        spawn_args.append(args)
        return _P()

    with patch(
        "jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec
    ):
        await StreamPlayer.spawn(input_args=("-f", "s16le"))

    argv = spawn_args[0]
    assert argv[0] == "ffplay"
    assert argv.index("-f") < argv.index("-i")


@pytest.mark.asyncio
async def test_play_raises_when_afplay_fails(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    async def _fake_exec(*args, **kwargs):
        class _P:
            returncode = 1

            async def communicate(self):
                return (b"", b"missing file")

            async def wait(self):
                return 1

        return _P()

    with patch("jarvis_cli.player.asyncio.create_subprocess_exec", side_effect=_fake_exec):
        with pytest.raises(RuntimeError):
            await play(audio)


# ---------------------------------------------------------------------------
# PCMPlayer — the in-process sounddevice sink. sounddevice is faked via
# sys.modules so these tests run (and mean the same thing) with or without
# the `audio` extra installed.
# ---------------------------------------------------------------------------

class _FakePortAudioError(Exception):
    pass


class _FakeRawStream:
    """Mimics callback-mode sounddevice.RawOutputStream: captures the
    callback so tests can pump it like PortAudio would, records
    start/stop/abort/close transitions."""

    def __init__(self, *, samplerate, channels, dtype, callback=None,
                 blocksize=None, latency=None):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.callback = callback
        self.blocksize = blocksize
        self.latency = latency
        self.started = False
        self.stopped = False
        self.closed = False
        self.aborted = False

    def start(self):
        self.started = True

    def pump(self, frames: int) -> bytes:
        """Invoke the callback the way PortAudio would; return what 'played'."""
        out = bytearray(frames * 2 * self.channels)
        self.callback(out, frames, None, None)
        return bytes(out)

    def stop(self):
        self.stopped = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


def _fake_sounddevice(created: list[_FakeRawStream]) -> types.ModuleType:
    mod = types.ModuleType("sounddevice")

    def _raw_output_stream(**kwargs):
        stream = _FakeRawStream(**kwargs)
        created.append(stream)
        return stream

    mod.RawOutputStream = _raw_output_stream
    mod.PortAudioError = _FakePortAudioError
    return mod


@pytest.mark.asyncio
async def test_pcm_player_spawn_opens_stream_with_pcm_spec():
    """spawn() must open (but NOT start) an int16 callback RawOutputStream
    with the provider's rate/channels and hand the player itself to on_spawn
    (the daemon's cancel-registration hook). Start is deferred until the
    prebuffer fills so the callback never begins by starving."""
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    seen = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1, on_spawn=seen.append)

    assert len(created) == 1
    stream = created[0]
    assert (stream.samplerate, stream.channels, stream.dtype) == (24000, 1, "int16")
    assert stream.callback is not None
    assert stream.started is False
    assert seen == [player]


@pytest.mark.asyncio
async def test_pcm_player_starts_only_after_prebuffer():
    """Playback must not start until PREBUFFER_SECONDS of audio is queued —
    the lead is what rides out sentence-piece prefill stalls."""
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        prebuffer_bytes = int(24000 * 2 * PCMPlayer.PREBUFFER_SECONDS)

        async def _small():
            yield b"\x01" * (prebuffer_bytes // 2)

        await player.feed(_small())
        assert created[0].started is False  # below threshold: keep buffering

        async def _rest():
            yield b"\x02" * (prebuffer_bytes // 2)

        await player.feed(_rest())
        assert created[0].started is True  # threshold crossed: rolling


@pytest.mark.asyncio
async def test_pcm_player_close_drains_via_callback_and_pads_underrun():
    """close() force-starts short utterances, waits for the callback to
    drain the ring buffer, then stops. A starving callback pads with silence
    (zeros) — never raises, never clicks."""
    from jarvis_cli.player import PCMPlayer

    async def _chunks():
        yield b"\x07\x07"
        yield b""  # empty chunks are skipped
        yield b"\x09\x09"

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        await player.feed(_chunks())
        stream = created[0]
        assert stream.started is False  # 4 bytes ≪ prebuffer

        close_task = asyncio.create_task(player.close())
        await asyncio.sleep(0.01)  # let close() force-start the stream
        assert stream.started is True
        played = stream.pump(2)  # PortAudio pulls 2 frames = 4 bytes
        await close_task

    assert played == b"\x07\x07\x09\x09"
    assert stream.stopped is True
    assert stream.closed is True
    # Buffer now empty: further pulls get pure silence, zero-padded.
    assert stream.pump(3) == b"\x00" * 6


@pytest.mark.asyncio
async def test_pcm_player_kill_aborts_and_close_stays_quiet():
    """kill() is the daemon's synchronous cancel hook: discard buffered audio,
    never raise (even twice); a close() after kill concludes quietly — the
    playback was cancelled, not failed."""
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)

        async def _chunks():
            yield b"\x05\x05" * 100

        await player.feed(_chunks())
        player.kill()
        player.kill()  # idempotent, still no raise
        await player.close()  # must not raise

    stream = created[0]
    assert stream.aborted is True
    assert stream.closed is True
    # No StopStream drain after a kill — the whole point is instant silence.
    assert stream.stopped is False
    # Ring buffer discarded: nothing left to play.
    assert player._buffered_seconds() == 0


@pytest.mark.asyncio
async def test_pcm_player_kill_survives_wedged_device():
    """The 2026-07-06 regression: PortAudio's stop path can block forever on
    a CoreAudio mutex (AudioOutputUnitStop vs the render callback's GIL
    wait). kill() fires on the daemon's cancel path — the event loop — so it
    must return immediately anyway, and close() must give up after
    RELEASE_TIMEOUT_SECONDS instead of freezing the speech queue with it."""
    import jarvis_cli.player as player_mod
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    unwedge = threading.Event()
    # The leak flips the process-wide wedge flag — keep that contained here.
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}), \
            patch.object(player_mod, "_pcm_wedged", False):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        stream = created[0]
        stream.abort = unwedge.wait  # abort now blocks like the real deadlock

        t0 = time.monotonic()
        player.kill()
        assert time.monotonic() - t0 < 0.5  # returned, loop never blocked

        with patch.object(PCMPlayer, "RELEASE_TIMEOUT_SECONDS", 0.2):
            await player.close()  # bounded wait, then leaks the stream

    assert stream.closed is False  # release never got past the wedged abort
    unwedge.set()  # let the leaked release thread exit


@pytest.mark.asyncio
async def test_pcm_release_leak_wedges_process_and_open_pcm_sink_falls_back():
    """The 2026-07-17 / 2026-08-15 regression: after close() leaks a stuck
    release thread, that thread may still hold CoreAudio's HAL mutex — so the
    NEXT Pa_OpenStream blocks forever and the speech queue silently fills for
    days. Once a release leaks, open_pcm_sink must never touch PortAudio
    again in this process: ffplay (its own process, its own HAL) still
    plays."""
    import jarvis_cli.player as player_mod
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    unwedge = threading.Event()
    fallback = object()

    class _FakeStreamPlayer:
        @classmethod
        async def spawn(cls, *, input_args=None, on_spawn=None):
            return fallback

    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}), \
            patch.object(player_mod, "_pcm_wedged", False), \
            patch("jarvis_cli.player.StreamPlayer", _FakeStreamPlayer):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        stream = created[0]
        stream.abort = unwedge.wait  # release wedges like the real deadlock

        player.kill()
        with patch.object(PCMPlayer, "RELEASE_TIMEOUT_SECONDS", 0.2):
            await player.close()  # bounded wait, leaks the release thread
        assert player_mod._pcm_wedged is True

        sink = await player_mod.open_pcm_sink(rate=24000, channels=1)

    assert sink is fallback
    assert len(created) == 1  # PortAudio was never opened again
    unwedge.set()  # let the leaked release thread exit


@pytest.mark.asyncio
async def test_pcm_player_spawn_bounded_when_open_wedges():
    """Pa_OpenStream itself can block forever on the wedged HAL mutex. spawn()
    must give up after OPEN_TIMEOUT_SECONDS (raising so the caller falls back
    to ffplay), mark the process wedged — and when the stuck open eventually
    completes, release the never-used stream instead of leaking it open."""
    import jarvis_cli.player as player_mod
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    unwedge = threading.Event()
    mod = types.ModuleType("sounddevice")

    def _blocked_stream(**kwargs):
        unwedge.wait()  # constructor blocks like Pa_OpenStream on the mutex
        stream = _FakeRawStream(**kwargs)
        created.append(stream)
        return stream

    mod.RawOutputStream = _blocked_stream
    mod.PortAudioError = _FakePortAudioError

    with patch.dict(sys.modules, {"sounddevice": mod}), \
            patch.object(PCMPlayer, "OPEN_TIMEOUT_SECONDS", 0.2), \
            patch.object(player_mod, "_pcm_wedged", False):
        with pytest.raises(RuntimeError, match="open stuck"):
            await PCMPlayer.spawn(rate=24000, channels=1)
        assert player_mod._pcm_wedged is True

        # The wedge clears later: the abandoned open must clean up after
        # itself rather than leave an untracked open stream behind.
        unwedge.set()
        for _ in range(100):
            if created and created[0].closed:
                break
            await asyncio.sleep(0.01)

    assert created and created[0].aborted is True and created[0].closed is True


@pytest.mark.asyncio
async def test_pcm_player_feed_after_kill_propagates():
    """After an external kill() feed must raise, and the daemon lets that
    propagate — the error is how it distinguishes cancel from TTS failure
    (mirroring ffplay's broken pipe)."""
    from jarvis_cli.player import PCMPlayer

    async def _chunks():
        yield b"too late"

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        player.kill()
        with pytest.raises(RuntimeError, match="cancelled"):
            await player.feed(_chunks())


@pytest.mark.asyncio
async def test_pcm_player_abort_never_raises():
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        await player.abort()  # must not raise

    assert created[0].aborted is True
    assert created[0].closed is True


@pytest.mark.asyncio
async def test_open_pcm_sink_prefers_pcm_player():
    from jarvis_cli.player import PCMPlayer, open_pcm_sink

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        sink = await open_pcm_sink(rate=24000, channels=1)

    assert isinstance(sink, PCMPlayer)
    assert created[0].callback is not None


@pytest.mark.asyncio
async def test_open_pcm_sink_falls_back_to_ffplay_and_warns_once():
    """Without sounddevice the sink must degrade to the ffplay StreamPlayer
    (spawned with the provider's decode flags) — and nag exactly once per
    process, not per utterance."""
    import jarvis_cli.player as player_mod

    spawn_calls: list[tuple] = []
    fallback = object()

    class _FakeStreamPlayer:
        @classmethod
        async def spawn(cls, *, input_args=None, on_spawn=None):
            spawn_calls.append((input_args, on_spawn))
            return fallback

    # sys.modules[name] = None makes `import sounddevice` raise ImportError.
    with patch.dict(sys.modules, {"sounddevice": None}), \
            patch("jarvis_cli.player.StreamPlayer", _FakeStreamPlayer), \
            patch("jarvis_cli.player.logger") as fake_logger, \
            patch.object(player_mod, "_pcm_fallback_warned", False):
        on_spawn = MagicMock()
        sink1 = await player_mod.open_pcm_sink(
            rate=24000, channels=1,
            input_args=("-f", "s16le"), on_spawn=on_spawn,
        )
        sink2 = await player_mod.open_pcm_sink(rate=24000, channels=1)

    assert sink1 is fallback and sink2 is fallback
    assert spawn_calls[0] == (("-f", "s16le"), on_spawn)
    fake_logger.warning.assert_called_once()
