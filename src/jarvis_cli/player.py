"""Audio playback helpers.

`play(path)`            — afplay; reads a finished audio file from disk.
`play_stream(chunks)`   — ffplay; reads MP3 chunks from stdin so playback
                          starts before synthesis completes.
`StreamPlayer`          — the underlying one-process session; lets a caller
                          feed several chunk iterators (e.g. per-sentence TTS
                          streams) through a single ffplay pipe gaplessly.

All accept an optional `on_spawn(proc)` callback so callers (the daemon
worker) can capture the subprocess handle for external cancellation.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path


async def play(
    audio: Path,
    *,
    on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "afplay", str(audio),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    if on_spawn is not None:
        on_spawn(proc)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"afplay failed: {err.decode(errors='replace')}")


class StreamPlayer:
    """One ffplay process fed incrementally — possibly across multiple chunk
    iterators (e.g. per-sentence TTS streams) — so consecutive sentences play
    gaplessly through a single pipe. close() ends stdin so ffplay drains its
    buffer and exits."""

    def __init__(self, proc: asyncio.subprocess.Process) -> None:
        self._proc = proc

    @classmethod
    async def spawn(
        cls,
        *,
        input_args: Sequence[str] | None = None,
        on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
    ) -> "StreamPlayer":
        """Start ffplay reading from stdin.

        ffplay buffers internally and starts playback once it has enough
        audio for a frame, so first sound usually plays within a few hundred
        ms of the first chunk — long before synthesis completes.

        By default ffplay auto-detects the container (works for MP3 from
        ElevenLabs). Providers that stream a headerless format — e.g. XTTS,
        which yields raw little-endian 16-bit PCM — pass `input_args` like
        `("-f", "s16le", "-ar", "24000", "-ac", "1")` so ffplay knows how to
        decode the byte stream.
        """
        proc = await asyncio.create_subprocess_exec(
            "ffplay",
            "-loglevel", "error",
            "-nodisp",
            "-autoexit",
            *(input_args or ()),
            "-i", "pipe:0",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdin is not None  # PIPE was requested above
        if on_spawn is not None:
            on_spawn(proc)
        return cls(proc)

    async def feed(self, chunks: AsyncIterator[bytes]) -> None:
        """Write one chunk iterator into the pipe. May be called repeatedly —
        each call appends to the same stream. Write errors (BrokenPipeError
        after an external kill, etc.) propagate so the caller can decide
        whether the death was a cancel or a genuine failure."""
        stdin = self._proc.stdin
        assert stdin is not None  # spawn() guarantees a PIPE
        async for chunk in chunks:
            if not chunk:
                continue
            stdin.write(chunk)
            await stdin.drain()

    async def close(self) -> None:
        """End stdin so ffplay drains its buffer and exits, then reap it.

        Raises RuntimeError on a nonzero exit — which includes the -9 an
        external cancel-kill produces; callers with a cancel concept must
        map that to "playback already concluded" themselves."""
        stdin = self._proc.stdin
        try:
            if stdin is not None and not stdin.is_closing():
                stdin.close()
                await stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass
        rc = await self._proc.wait()
        if rc != 0:
            # Surface ffplay's own complaint (bad option, no audio device, …)
            # — "exited with code 1" alone cost us a debugging session once.
            detail = ""
            if self._proc.stderr is not None:
                err = await self._proc.stderr.read()
                if err:
                    detail = f": {err.decode(errors='replace').strip()[:200]}"
            raise RuntimeError(f"ffplay exited with code {rc}{detail}")

    async def abort(self) -> None:
        """Kill the process and reap it. Never raises — this is the cleanup
        for error/fallback paths, where a second exception would only mask
        the one being handled."""
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        await self._proc.wait()


async def play_stream(
    chunks: AsyncIterator[bytes],
    *,
    on_spawn: Callable[[asyncio.subprocess.Process], None] | None = None,
    input_args: Sequence[str] | None = None,
) -> None:
    """Spawn ffplay reading audio from stdin and feed it chunks as they arrive.

    One-shot convenience over StreamPlayer: spawn → feed → close, with close
    always running so ffplay is never orphaned even when the chunk iterator
    or a write raises. See StreamPlayer.spawn for the `input_args` contract.
    """
    player = await StreamPlayer.spawn(input_args=input_args, on_spawn=on_spawn)
    try:
        await player.feed(chunks)
    except BaseException:
        # Feed failed: still conclude the process so nothing is orphaned, but
        # let the original error propagate — a secondary nonzero-exit raised
        # by close() would only mask the real cause.
        with contextlib.suppress(Exception):
            await player.close()
        raise
    await player.close()
