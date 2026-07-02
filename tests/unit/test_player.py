import sys
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
    """Mimics sounddevice.RawOutputStream: write blocks (here: records),
    abort discards, and writing to an aborted/closed stream raises — the
    behaviour PCMPlayer's cancel detection relies on."""

    def __init__(self, *, samplerate, channels, dtype):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.started = False
        self.stopped = False
        self.closed = False
        self.aborted = False
        self.written: list[bytes] = []

    def start(self):
        self.started = True

    def write(self, data):
        if self.aborted or self.closed:
            raise _FakePortAudioError("stream is stopped")
        self.written.append(bytes(data))

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
    """spawn() must open a started int16 RawOutputStream with the provider's
    rate/channels and hand the player itself to on_spawn (the daemon's
    cancel-registration hook)."""
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    seen = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1, on_spawn=seen.append)

    assert len(created) == 1
    stream = created[0]
    assert (stream.samplerate, stream.channels, stream.dtype) == (24000, 1, "int16")
    assert stream.started is True
    assert seen == [player]


@pytest.mark.asyncio
async def test_pcm_player_feeds_chunks_and_close_drains():
    """feed() writes each nonempty chunk; close() stops (PortAudio drains
    buffered audio) then releases the device."""
    from jarvis_cli.player import PCMPlayer

    async def _chunks():
        yield b"chunk-A"
        yield b""  # empty chunks are skipped, not written
        yield b"chunk-B"

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        await player.feed(_chunks())
        await player.close()

    stream = created[0]
    assert stream.written == [b"chunk-A", b"chunk-B"]
    assert stream.stopped is True
    assert stream.closed is True


@pytest.mark.asyncio
async def test_pcm_player_kill_aborts_and_close_stays_quiet():
    """kill() is the daemon's synchronous cancel hook: discard buffered audio,
    never raise (even twice); a close() after kill concludes quietly — the
    playback was cancelled, not failed."""
    from jarvis_cli.player import PCMPlayer

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        player.kill()
        player.kill()  # idempotent, still no raise
        await player.close()  # must not raise

    stream = created[0]
    assert stream.aborted is True
    assert stream.closed is True
    # No StopStream drain after a kill — the whole point is instant silence.
    assert stream.stopped is False


@pytest.mark.asyncio
async def test_pcm_player_feed_after_kill_propagates():
    """After an external kill() the aborted stream raises on write, and feed
    must let it propagate — that error is how the daemon distinguishes
    cancel from TTS failure (mirroring ffplay's broken pipe)."""
    from jarvis_cli.player import PCMPlayer

    async def _chunks():
        yield b"too late"

    created: list[_FakeRawStream] = []
    with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
        player = await PCMPlayer.spawn(rate=24000, channels=1)
        player.kill()
        with pytest.raises(_FakePortAudioError):
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
    assert created[0].started is True


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
