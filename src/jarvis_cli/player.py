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
import threading
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
        # Surface ffplay's own complaints even on rc=0 — SDL can fail to open
        # an audio device, warn on stderr, and still exit clean while playing
        # to nowhere. "exited with code 1" alone cost us a debugging session
        # once; silent-success stderr cost us another.
        detail = ""
        stderr = getattr(self._proc, "stderr", None)  # test fakes omit it
        if stderr is not None:
            err = await stderr.read()
            if err:
                detail = f": {err.decode(errors='replace').strip()[:200]}"
        if rc != 0:
            raise RuntimeError(f"ffplay exited with code {rc}{detail}")
        if detail:
            logger.debug("ffplay exited 0 with stderr{}", detail)

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
    discard instead of a SIGKILL, no CLI-flag surface to get wrong, and —
    decisively — PortAudio binds the real default output device, where
    daemon-spawned SDL was intermittently landing on an inaudible one.

    Jitter-buffered: the PortAudio callback pulls from a Python-side ring
    buffer and emits clean silence when it runs dry, so a synthesis stall
    (the GPT prefill between sentence pieces produces nothing for up to a
    second) sounds like a natural pause instead of a glitch. Playback only
    starts once `prebuffer_seconds` of audio is queued (or the stream is
    closed, for utterances shorter than that), building enough lead to ride
    out those stalls entirely at RTF < 1.
    """

    #: Audio queued before the device starts consuming. Chosen against the
    #: measured worst stall: a sentence-piece prefill (~0.5-1s of silence
    #: from the decoder) versus decode lead accumulating at ~0.2s per chunk.
    PREBUFFER_SECONDS = 1.0

    #: How long close() waits for the release thread before giving up on the
    #: device. Past this we leak the stream rather than stall the speech
    #: queue — the release can genuinely never return (see _release).
    RELEASE_TIMEOUT_SECONDS = 5.0

    def __init__(self, stream: Any, rate: int, channels: int) -> None:
        self._stream = stream  # a sounddevice.RawOutputStream (callback mode)
        self._rate = rate
        self._bytes_per_frame = 2 * channels  # int16
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._started = False
        self._killed = False
        self._eof = False
        # Terminal PortAudio ops (stop/abort/close) run exactly once, on this
        # dedicated thread — see _release for why they can never run on the
        # event loop.
        self._release_thread: threading.Thread | None = None
        # Diagnostics: how often the callback ran vs ran dry — logged at
        # close() so stutter reports can be pinned to starvation (or ruled
        # out) from the daemon log alone.
        self._callbacks = 0
        self._underruns = 0

    @classmethod
    async def spawn(
        cls,
        *,
        rate: int,
        channels: int,
        on_spawn: Callable[[Cancellable], None] | None = None,
    ) -> "PCMPlayer":
        """Open a PortAudio output stream for headerless int16 PCM.

        sounddevice is imported lazily so the daemon stays importable without
        the `audio` extra; import errors and device-open failures propagate —
        `open_pcm_sink` maps them to the ffplay fallback. The stream is NOT
        started here: it starts once the prebuffer fills (see feed) so the
        callback never begins by starving.
        """

        def _open() -> "PCMPlayer":
            import sounddevice as sd  # noqa: PLC0415 — optional extra

            player_box: list[PCMPlayer] = []

            def _callback(outdata: Any, frames: int, _time: Any, _status: Any) -> None:
                player = player_box[0]
                need = frames * player._bytes_per_frame
                with player._lock:
                    take = min(need, len(player._buf))
                    outdata[:take] = bytes(player._buf[:take])
                    del player._buf[:take]
                player._callbacks += 1
                # Ran dry mid-utterance: pad with silence. No exception, no
                # click — the deficit is repaid when the decoder catches up.
                if take < need:
                    player._underruns += 1
                    outdata[take:need] = bytes(need - take)

            stream = sd.RawOutputStream(
                samplerate=rate, channels=channels, dtype="int16",
                callback=_callback,
                # The callback is Python, so it contends for the GIL with the
                # decode thread. At the default ~10ms block it fires ~100x/s
                # and any GIL stall tears the audio (audible crackle). 200ms
                # blocks give each callback a deadline no Python stall
                # plausibly misses, at the cost of latency we already spend
                # on the prebuffer anyway.
                blocksize=rate // 5,
                latency="high",
            )
            player = cls(stream, rate, channels)
            player_box.append(player)
            return player

        player = await asyncio.to_thread(_open)
        if on_spawn is not None:
            on_spawn(player)
        return player

    def _buffered_seconds(self) -> float:
        with self._lock:
            return len(self._buf) / (self._rate * self._bytes_per_frame)

    def _start_if_ready(self, *, force: bool = False) -> None:
        if self._started or self._killed:
            return
        if force or self._buffered_seconds() >= self.PREBUFFER_SECONDS:
            self._started = True
            self._stream.start()

    async def feed(self, chunks: AsyncIterator[bytes]) -> None:
        """Append one chunk iterator to the ring buffer. May be called
        repeatedly — consecutive sentences play gaplessly. After an external
        kill() this raises, and that propagating error is how the daemon
        detects a cancel (mirroring ffplay's broken pipe), so it must NOT be
        swallowed here."""
        async for chunk in chunks:
            if self._killed:
                raise RuntimeError("PCM playback cancelled")
            if not chunk:
                continue
            with self._lock:
                self._buf.extend(chunk)
            await asyncio.to_thread(self._start_if_ready)

    def _release(self, *, abort: bool) -> None:
        """Stop (or abort) and close the stream. Runs on _release_thread,
        NEVER on the event loop: PortAudio's stop path takes a CoreAudio
        mutex the render callback can hold while waiting for the GIL, and
        that wait is unbounded — an inline call froze the whole daemon for
        six hours on 2026-07-06 (loop dead, socket backlog full, SIGTERM
        unserviceable)."""
        with contextlib.suppress(Exception):
            if abort:
                self._stream.abort()
            else:
                self._stream.stop()
            self._stream.close()

    def _start_release(self, *, abort: bool) -> threading.Thread:
        """Kick off (or return the already-running) release thread. First
        caller wins abort-vs-stop; the ops must not race each other on one
        stream."""
        with self._lock:
            if self._release_thread is None:
                self._release_thread = threading.Thread(
                    target=self._release, kwargs={"abort": abort}, daemon=True,
                )
                self._release_thread.start()
            return self._release_thread

    async def _join_release(self, thread: threading.Thread) -> None:
        await asyncio.to_thread(thread.join, self.RELEASE_TIMEOUT_SECONDS)
        if thread.is_alive():
            logger.warning(
                "pcm sink: device release stuck past {}s — leaking the "
                "stream and moving on", self.RELEASE_TIMEOUT_SECONDS,
            )

    async def close(self) -> None:
        """Play out whatever is buffered, then release the device. After
        kill() this returns without raising — a cancelled playback is
        concluded, not failed."""
        self._eof = True
        if self._killed:
            # kill() already launched the abort; just bound the wait.
            await self._join_release(self._start_release(abort=True))
            return
        # An utterance shorter than the prebuffer never hit the start
        # threshold — start now so it still plays.
        await asyncio.to_thread(self._start_if_ready, force=True)
        while not self._killed and self._buffered_seconds() > 0:
            await asyncio.sleep(0.05)
        # Let the device drain its own last callback buffer before stopping.
        await asyncio.sleep(0.1)
        await self._join_release(self._start_release(abort=self._killed))
        # The final callback legitimately runs short (end of utterance), so
        # one underrun is expected; more means playback starved mid-stream.
        logger.debug(
            "pcm sink: {} callbacks, {} underruns", self._callbacks, self._underruns,
        )

    def kill(self) -> None:
        """Synchronous cancel hook (the `Cancellable` contract): discard
        buffered audio immediately. Never raises, never blocks — the device
        release happens on _release_thread."""
        self._killed = True
        with self._lock:
            self._buf.clear()
        self._start_release(abort=True)

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
