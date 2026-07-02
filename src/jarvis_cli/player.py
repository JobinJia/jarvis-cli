"""Audio playback helpers.

`play(path)`            — afplay; reads a finished audio file from disk.
`play_stream(chunks)`   — ffplay; reads MP3 chunks from stdin so playback
                          starts before synthesis completes.
`StreamPlayer`          — the underlying one-process session; lets a caller
                          feed several chunk iterators (e.g. per-sentence TTS
                          streams) through a single ffplay pipe gaplessly.
`PCMPlayer`             — in-process raw-PCM sink via sounddevice/PortAudio;
                          same feed/close/abort surface as StreamPlayer but
                          no subprocess at all.
`open_pcm_sink(...)`    — PCMPlayer when sounddevice works, StreamPlayer
                          (ffplay) otherwise.

All accept an optional `on_spawn(handle)` callback so callers (the daemon
worker) can capture the playback handle for external cancellation — see
`Cancellable` for the contract that handle satisfies.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from loguru import logger


class Cancellable(Protocol):
    """What the daemon's cancel path needs from an in-flight playback handle:
    a synchronous kill that silences audio immediately. Satisfied structurally
    by `asyncio.subprocess.Process` (SIGKILL, may raise ProcessLookupError)
    and implemented directly by `PCMPlayer` (buffer discard, never raises)."""

    def kill(self) -> None: ...


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


class PCMPlayer:
    """In-process raw-PCM sink via sounddevice/PortAudio.

    Duck-type-compatible with StreamPlayer (feed/close/abort) so the daemon's
    streaming paths can hold either. Versus ffplay: no ~100-150ms process
    spawn + SDL device open per utterance, cancel is an instant buffer
    discard instead of a SIGKILL, and there's no CLI-flag surface to get
    wrong (the `-ac`-doesn't-exist class of bug).
    """

    def __init__(self, stream: Any) -> None:  # a sounddevice.RawOutputStream
        self._stream = stream
        self._killed = False

    @classmethod
    async def spawn(
        cls,
        *,
        rate: int,
        channels: int,
        on_spawn: Callable[[Cancellable], None] | None = None,
    ) -> "PCMPlayer":
        """Open + start a PortAudio output stream for headerless int16 PCM.

        sounddevice is imported lazily so the daemon stays importable without
        the `audio` extra; import errors and device-open failures propagate —
        `open_pcm_sink` maps them to the ffplay fallback. Open + start run
        off the event loop because PortAudio device open can block ~tens of
        ms (still an order of magnitude under an ffplay spawn).
        """

        def _open() -> Any:
            import sounddevice as sd  # noqa: PLC0415 — optional extra

            stream = sd.RawOutputStream(
                samplerate=rate, channels=channels, dtype="int16",
            )
            stream.start()
            return stream

        player = cls(await asyncio.to_thread(_open))
        if on_spawn is not None:
            on_spawn(player)
        return player

    async def feed(self, chunks: AsyncIterator[bytes]) -> None:
        """Write one chunk iterator into the device. May be called repeatedly —
        each call appends to the same stream, so consecutive sentences play
        gaplessly. Write errors propagate — after an external kill() the
        aborted stream raises on write, and that propagating error is how the
        daemon detects a cancel (mirroring ffplay's broken pipe), so it must
        NOT be swallowed here."""
        async for chunk in chunks:
            if not chunk:
                continue
            # RawOutputStream.write accepts bytes and blocks until the chunk
            # fits in PortAudio's buffer — that blocking IS our backpressure,
            # so it runs in a worker thread to keep the event loop live.
            await asyncio.to_thread(self._stream.write, chunk)

    async def close(self) -> None:
        """Drain buffered audio (PortAudio StopStream plays out what's queued)
        and release the device. After kill() this returns without raising —
        a cancelled playback is concluded, not failed."""
        if self._killed:
            with contextlib.suppress(Exception):
                self._stream.close()
            return
        await asyncio.to_thread(self._stream.stop)
        self._stream.close()

    def kill(self) -> None:
        """Synchronous cancel hook (the `Cancellable` contract): discard
        buffered audio immediately. Never raises — the daemon's cancel path
        only guards against ProcessLookupError."""
        self._killed = True
        with contextlib.suppress(Exception):
            self._stream.abort()

    async def abort(self) -> None:
        """Kill + release the device. Never raises — this is the cleanup for
        error/fallback paths, where a second exception would only mask the
        one being handled."""
        self.kill()
        await self.close()


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


# One warning per process: whether sounddevice is installed/usable is decided
# at daemon start and doesn't change — repeating it per utterance is spam.
_pcm_fallback_warned = False


async def open_pcm_sink(
    *,
    rate: int,
    channels: int,
    input_args: Sequence[str] | None = None,
    on_spawn: Callable[[Cancellable], None] | None = None,
) -> "PCMPlayer | StreamPlayer":
    """Best sink for a raw-PCM stream: in-process PCMPlayer when sounddevice
    is available, else the ffplay StreamPlayer (spawned with `input_args`,
    the provider's decode flags). Both expose feed/close/abort, so callers
    don't care which they got."""
    global _pcm_fallback_warned
    try:
        return await PCMPlayer.spawn(rate=rate, channels=channels, on_spawn=on_spawn)
    except Exception as exc:  # ImportError or PortAudio init failure
        if not _pcm_fallback_warned:
            _pcm_fallback_warned = True
            logger.warning(
                "sounddevice unavailable ({}); falling back to ffplay", exc,
            )
        return await StreamPlayer.spawn(input_args=input_args, on_spawn=on_spawn)
