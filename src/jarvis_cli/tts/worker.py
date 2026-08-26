"""Synthesis worker: one provider, one child process, spoken to over pipes.

Why this exists: XTTS leaks ~40 MB of native memory per utterance (see
factory.HEAVY_PROVIDERS for the measurement), and native memory only comes back
when the process ends. Ending the DAEMON to reclaim it would take the socket,
the event queue and the retrieval index down with it, so the model runs out
here instead — this process is disposable, and `worker_client` replaces it once
it has leaked enough.

Requests arrive as JSON lines on stdin, replies as protocol frames on stdout.
Logs go to stderr, which the daemon inherits, so worker lines land in the same
daemon log file as everything else.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from ..config import DEFAULT_CONFIG_PATH, load_config
from .factory import make_provider
from .protocol import write_frame


async def _stdio() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """asyncio streams over this process's stdin/stdout.

    Both directions must be non-blocking: a synthesized piece can be megabytes,
    and a blocking write would stall the loop exactly when an abort needs to be
    read.
    """
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin,
    )
    transport, protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout,
    )
    return reader, asyncio.StreamWriter(transport, protocol, None, loop)


class _Worker:
    def __init__(self, provider: Any, writer: asyncio.StreamWriter) -> None:
        self._provider = provider
        self._writer = writer
        # At most one synthesis runs at a time — the daemon speaks one
        # utterance at a time, and serializing here keeps peak memory to a
        # single inference rather than however many requests happen to overlap.
        self._current: asyncio.Task | None = None
        self._current_id: int | None = None

    async def dispatch(self, req: dict[str, Any]) -> None:
        op, rid = req.get("op"), int(req.get("id", 0))
        if op == "abort":
            # Cancelling the task closes the provider's async generator, whose
            # `finally` sets the stop flag the decoder polls between pieces —
            # the same path an in-process cancel used to take.
            if self._current is not None and self._current_id == rid:
                self._current.cancel()
            return
        if self._current is not None and not self._current.done():
            await self._fail(rid, "worker busy")
            return
        self._current_id = rid
        self._current = asyncio.create_task(self._run(op, rid, req))

    async def _run(self, op: str | None, rid: int, req: dict[str, Any]) -> None:
        try:
            if op == "prewarm":
                await self._provider.prewarm()
                await write_frame(self._writer, {"id": rid, "type": "done"})
            elif op == "stream":
                await self._stream(rid, req)
            elif op == "synthesize":
                out = await self._provider.synthesize(
                    req["text"], req["lang"], Path(req["out_path"]),
                    voice_id=req.get("voice_id"), emotion=req.get("emotion"),
                )
                await write_frame(
                    self._writer, {"id": rid, "type": "done", "path": str(out)},
                )
            else:
                await self._fail(rid, f"unknown op {op!r}")
        except asyncio.CancelledError:
            # An abort, not a fault. The daemon already stopped listening for
            # this id; it just needs the worker to stay usable.
            with contextlib.suppress(Exception):
                await write_frame(self._writer, {"id": rid, "type": "aborted"})
        except Exception as exc:  # noqa: BLE001 — every failure is reportable
            await self._fail(rid, f"{type(exc).__name__}: {exc}")

    async def _stream(self, rid: int, req: dict[str, Any]) -> None:
        chunks = self._provider.stream(
            req["text"], req["lang"],
            voice_id=req.get("voice_id"), emotion=req.get("emotion"),
        )
        try:
            async for chunk in chunks:
                if chunk:
                    await write_frame(
                        self._writer, {"id": rid, "type": "chunk"}, chunk,
                    )
        finally:
            # Closing the generator is what signals the decoder to stop when
            # this task is cancelled mid-utterance.
            with contextlib.suppress(Exception):
                await chunks.aclose()
        await write_frame(self._writer, {"id": rid, "type": "done"})

    async def _fail(self, rid: int, message: str) -> None:
        logger.warning("tts-worker: request {} failed: {}", rid, message)
        with contextlib.suppress(Exception):
            await write_frame(
                self._writer, {"id": rid, "type": "error", "message": message},
            )


async def _serve(provider_name: str, config_path: str) -> int:
    cfg = load_config(Path(config_path))
    provider = make_provider(provider_name, cfg)
    if provider is None:
        logger.error("tts-worker: unknown provider {!r}", provider_name)
        return 2
    reader, writer = await _stdio()
    worker = _Worker(provider, writer)
    logger.info("tts-worker: serving {}", provider_name)
    while True:
        line = await reader.readline()
        if not line:  # daemon closed the pipe — recycle or shutdown
            break
        try:
            import json  # noqa: PLC0415 — only needed on the request path
            req = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            logger.warning("tts-worker: dropped malformed request")
            continue
        await worker.dispatch(req)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis-tts-worker")
    ap.add_argument("--provider", required=True)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)
    try:
        return asyncio.run(_serve(args.provider, args.config))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
