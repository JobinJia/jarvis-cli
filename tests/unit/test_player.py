from pathlib import Path
from unittest.mock import patch

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
