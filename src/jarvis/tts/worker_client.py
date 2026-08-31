"""Daemon-side handle on a synthesis worker child (see `worker`).

Satisfies `TTSProvider`, so nothing in the daemon's playback path has to know
whether the model lives in this process or the next one over. What it adds is a
`recycle()` the daemon calls at an idle moment: the child is replaced, its
leaked native memory returns to the OS with it, and no event, socket or index
is disturbed.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from loguru import logger

from ..types import Lang
from .factory import provider_class
from .protocol import read_frame
from .providers.base import TTSProvider


class WorkerProvider(TTSProvider):
    """Runs one provider in a child process and forwards calls to it.

    The child is spawned lazily and respawned after a crash, so a worker that
    dies takes one utterance with it (the daemon's existing fallback chain
    covers that) rather than silencing Jarvis until someone restarts him.
    """

    #: How long to wait for a child to load its model and answer `prewarm`.
    #: XTTS takes ~30s cold, and minutes when the machine is paging hard.
    PREWARM_TIMEOUT_SECONDS = 300.0
    #: Grace period between SIGTERM and SIGKILL when recycling.
    STOP_TIMEOUT_SECONDS = 5.0

    def __init__(
        self, provider_name: str, config_path: str | Path, max_syntheses: int,
    ) -> None:
        self.name = provider_name
        self._config_path = str(config_path)
        self._max_syntheses = max_syntheses
        # The streaming contract is a property of the provider CLASS, so it can
        # be read here without constructing anything heavy — the daemon needs
        # it (sink selection) before a child has even been spawned.
        cls = provider_class(provider_name)
        self.supports_streaming = getattr(cls, "supports_streaming", False)
        self.stream_input_args = getattr(cls, "stream_input_args", None)
        self.stream_pcm = getattr(cls, "stream_pcm", None)
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.StreamReader | None = None
        # One request in flight at a time: the daemon speaks one utterance at a
        # time, and the frames on the pipe are a single ordered stream.
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._syntheses = 0

    # ---- child lifecycle -------------------------------------------------

    @property
    def syntheses(self) -> int:
        return self._syntheses

    @property
    def should_recycle(self) -> bool:
        return self._max_syntheses > 0 and self._syntheses >= self._max_syntheses

    async def _ensure_child(self) -> asyncio.subprocess.Process:
        proc = self._proc
        if proc is not None and proc.returncode is None:
            return proc
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "jarvis.tts.worker",
            "--provider", self.name, "--config", self._config_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            # stderr is inherited on purpose: the child's loguru lines land in
            # the daemon's own log, where anyone debugging is already looking.
        )
        self._proc = proc
        self._reader = proc.stdout
        self._syntheses = 0
        logger.info("tts-worker: spawned {} (pid {})", self.name, proc.pid)
        return proc

    async def recycle(self) -> None:
        """Replace the child, returning its leaked memory to the OS.

        Callers must only invoke this when nothing is playing — it is the whole
        point of the design that the cost lands in a gap, not mid-sentence.
        """
        proc, self._proc, self._reader = self._proc, None, None
        if proc is None or proc.returncode is not None:
            return
        logger.info(
            "tts-worker: recycling {} (pid {}) after {} syntheses",
            self.name, proc.pid, self._syntheses,
        )
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), self.STOP_TIMEOUT_SECONDS)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        self._syntheses = 0

    async def aclose(self) -> None:
        """Daemon shutdown: take the child with us rather than orphan it."""
        await self.recycle()

    # ---- request plumbing ------------------------------------------------

    async def _send(self, req: dict) -> None:
        proc = await self._ensure_child()
        assert proc.stdin is not None  # PIPE requested in _ensure_child
        proc.stdin.write(json.dumps(req, ensure_ascii=False).encode() + b"\n")
        await proc.stdin.drain()

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    async def _abort(self, rid: int) -> None:
        """Best-effort: tell the child to stop decoding a line nobody will hear.
        A failure here is not worth surfacing — the child is about to be
        recycled or is already gone."""
        with contextlib.suppress(Exception):
            await self._send({"id": rid, "op": "abort"})

    def _child_died(self) -> RuntimeError:
        proc, self._proc, self._reader = self._proc, None, None
        rc = proc.returncode if proc else None
        return RuntimeError(f"tts worker {self.name} died (rc={rc})")

    # ---- TTSProvider surface --------------------------------------------

    async def prewarm(self) -> None:
        async with self._lock:
            rid = self._new_id()
            await self._send({"id": rid, "op": "prewarm"})
            await asyncio.wait_for(
                self._await_done(rid), self.PREWARM_TIMEOUT_SECONDS,
            )

    async def _await_done(self, rid: int) -> None:
        while True:
            reader = self._reader
            if reader is None:
                raise self._child_died()
            frame = await read_frame(reader)
            if frame is None:
                raise self._child_died()
            header, _ = frame
            if header.get("id") != rid:
                continue
            kind = header.get("type")
            if kind == "done":
                return
            if kind == "error":
                raise RuntimeError(header.get("message", "worker error"))
            if kind == "aborted":
                raise asyncio.CancelledError

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> Path:
        async with self._lock:
            rid = self._new_id()
            await self._send({
                "id": rid, "op": "synthesize", "text": text, "lang": lang,
                "voice_id": voice_id, "emotion": emotion,
                "out_path": str(out_path),
            })
            await self._await_done(rid)
            self._syntheses += 1
            return out_path

    async def stream(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
        emotion: str | None = None,
    ) -> AsyncIterator[bytes]:
        async with self._lock:
            rid = self._new_id()
            await self._send({
                "id": rid, "op": "stream", "text": text, "lang": lang,
                "voice_id": voice_id, "emotion": emotion,
            })
            delivered = False
            try:
                while True:
                    reader = self._reader
                    if reader is None:
                        raise self._child_died()
                    frame = await read_frame(reader)
                    if frame is None:
                        raise self._child_died()
                    header, payload = frame
                    if header.get("id") != rid:
                        continue
                    kind = header.get("type")
                    if kind == "chunk":
                        delivered = True
                        yield payload
                    elif kind == "done":
                        break
                    elif kind == "aborted":
                        break
                    elif kind == "error":
                        raise RuntimeError(header.get("message", "worker error"))
            except GeneratorExit:
                # The daemon stopped consuming — a cancel. Tell the child so it
                # stops decoding a line that will never be heard; do NOT wait
                # for it to acknowledge, the queue moves on immediately.
                await self._abort(rid)
                raise
            finally:
                if delivered:
                    self._syntheses += 1

    async def healthcheck(self) -> bool:
        proc = self._proc
        return proc is None or proc.returncode is None
