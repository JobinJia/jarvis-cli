"""Tiny HTTP /health server using stdlib asyncio.

We avoid a heavy web framework dep — this is a single-route status endpoint.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

from loguru import logger


class HealthServer:
    def __init__(
        self,
        host: str,
        port: int,
        state_getter: Callable[[], dict[str, Any]],
    ) -> None:
        self.host = host
        self.requested_port = port
        self.actual_port = port
        self._state_getter = state_getter
        self._server: asyncio.base_events.Server | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.actual_port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, host=self.host, port=self.requested_port
        )
        self.actual_port = self._server.sockets[0].getsockname()[1]
        logger.info("Health server on {}", self.url)

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            line = await reader.readline()
            # Consume remaining headers
            while True:
                hl = await reader.readline()
                if hl in (b"\r\n", b"\n", b""):
                    break
            if not line.startswith(b"GET /health"):
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            body = json.dumps(self._state_getter()).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            )
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
