import asyncio

import httpx
import pytest

from jarvis.daemon.health import HealthServer


@pytest.mark.asyncio
async def test_health_endpoint_returns_status():
    state = {"queue_size": 3, "dropped": 1, "last_text": "Sir, hello."}
    server = HealthServer(host="127.0.0.1", port=0, state_getter=lambda: state)
    await server.start()
    try:
        # trust_env=False: with macOS system proxy settings active, httpx
        # would otherwise route this loopback call through the proxy and the
        # test would exercise the proxy, not the server.
        async with httpx.AsyncClient(base_url=server.url, trust_env=False) as c:
            r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["queue_size"] == 3
        assert body["dropped"] == 1
        assert body["last_text"] == "Sir, hello."
    finally:
        await server.stop()


async def _raw_request(server: HealthServer, request: bytes) -> bytes:
    reader, writer = await asyncio.open_connection(server.host, server.actual_port)
    try:
        writer.write(request)
        await writer.drain()
        return await reader.read(4096)
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_health_accepts_absolute_form_request_target():
    """Proxied clients (macOS system proxy reaches even loopback traffic)
    send `GET http://host:port/health` — the server must serve that, not 404."""
    server = HealthServer(host="127.0.0.1", port=0, state_getter=lambda: {"ok": 1})
    await server.start()
    try:
        raw = await _raw_request(
            server,
            f"GET {server.url}/health HTTP/1.1\r\nHost: x\r\n\r\n".encode(),
        )
        assert raw.startswith(b"HTTP/1.1 200 OK")
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_health_unknown_path_is_404_in_both_forms():
    server = HealthServer(host="127.0.0.1", port=0, state_getter=lambda: {"ok": 1})
    await server.start()
    try:
        for target in ("/nothealth", f"{server.url}/nothealth"):
            raw = await _raw_request(
                server, f"GET {target} HTTP/1.1\r\nHost: x\r\n\r\n".encode(),
            )
            assert raw.startswith(b"HTTP/1.1 404"), target
    finally:
        await server.stop()
