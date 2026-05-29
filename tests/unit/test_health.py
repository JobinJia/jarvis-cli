import asyncio

import httpx
import pytest

from jarvis_cli.daemon.health import HealthServer


@pytest.mark.asyncio
async def test_health_endpoint_returns_status():
    state = {"queue_size": 3, "dropped": 1, "last_text": "Sir, hello."}
    server = HealthServer(host="127.0.0.1", port=0, state_getter=lambda: state)
    await server.start()
    try:
        async with httpx.AsyncClient(base_url=server.url) as c:
            r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["queue_size"] == 3
        assert body["dropped"] == 1
        assert body["last_text"] == "Sir, hello."
    finally:
        await server.stop()
