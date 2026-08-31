import json

import httpx
import pytest
import respx

from jarvis.config import WebhookConfig
from jarvis.notify import webhook
from jarvis.types import Event


def _event(**kw) -> Event:
    base = dict(
        notification_type="idle_prompt",
        tool_name="Bash",
        tool_input={},
        cwd="/proj",
        received_at=1718000000.0,
    )
    base.update(kw)
    return Event(**base)


@pytest.mark.asyncio
async def test_disabled_does_not_post():
    cfg = WebhookConfig(enabled=False, url="https://example.com/hook")
    with respx.mock(base_url="https://example.com", assert_all_called=False) as router:
        route = router.post("/hook").respond(200)
        ok = await webhook.notify(cfg, _event(), "hello")
    assert ok is False
    assert route.called is False


@pytest.mark.asyncio
async def test_enabled_but_empty_url_is_noop():
    cfg = WebhookConfig(enabled=True, url="")
    assert await webhook.notify(cfg, _event(), "hello") is False


@pytest.mark.asyncio
async def test_posts_expected_payload():
    cfg = WebhookConfig(enabled=True, url="https://example.com/hook")
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").mock(side_effect=_handler)
        ok = await webhook.notify(cfg, _event(), "Sir, the build is done.")

    assert ok is True
    assert captured == {
        "text": "Sir, the build is done.",
        "notification_type": "idle_prompt",
        "tool_name": "Bash",
        "cwd": "/proj",
        "received_at": 1718000000.0,
    }


@pytest.mark.asyncio
async def test_auth_header_injected_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_WEBHOOK_TOKEN", "secret-token")
    cfg = WebhookConfig(
        enabled=True,
        url="https://example.com/hook",
        headers={"X-Static": "v"},
        auth_header="Authorization",
        auth_env="MY_WEBHOOK_TOKEN",
    )
    seen: dict[str, str] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization", "")
        seen["static"] = request.headers.get("X-Static", "")
        return httpx.Response(200)

    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").mock(side_effect=_handler)
        await webhook.notify(cfg, _event(), "hi")

    assert seen["auth"] == "secret-token"
    assert seen["static"] == "v"


@pytest.mark.asyncio
async def test_auth_header_omitted_when_env_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MY_WEBHOOK_TOKEN", raising=False)
    cfg = WebhookConfig(
        enabled=True,
        url="https://example.com/hook",
        auth_header="Authorization",
        auth_env="MY_WEBHOOK_TOKEN",
    )
    seen: dict[str, bool] = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        seen["has_auth"] = "Authorization" in request.headers
        return httpx.Response(200)

    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").mock(side_effect=_handler)
        ok = await webhook.notify(cfg, _event(), "hi")

    assert ok is True
    assert seen["has_auth"] is False


@pytest.mark.asyncio
async def test_event_allowlist_filters():
    cfg = WebhookConfig(
        enabled=True,
        url="https://example.com/hook",
        events=["permission_prompt"],
    )
    with respx.mock(base_url="https://example.com", assert_all_called=False) as router:
        route = router.post("/hook").respond(200)
        ok = await webhook.notify(cfg, _event(notification_type="idle_prompt"), "hi")
    assert ok is False
    assert route.called is False


@pytest.mark.asyncio
async def test_allowlist_passes_matching_type():
    cfg = WebhookConfig(
        enabled=True,
        url="https://example.com/hook",
        events=["idle_prompt"],
    )
    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").respond(200)
        ok = await webhook.notify(cfg, _event(notification_type="idle_prompt"), "hi")
    assert ok is True


@pytest.mark.asyncio
async def test_non_2xx_fails_soft():
    cfg = WebhookConfig(enabled=True, url="https://example.com/hook")
    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").respond(500)
        ok = await webhook.notify(cfg, _event(), "hi")
    assert ok is False  # logged + swallowed, no raise


@pytest.mark.asyncio
async def test_connection_error_fails_soft():
    cfg = WebhookConfig(enabled=True, url="https://example.com/hook")
    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").mock(
            side_effect=httpx.ConnectError("boom")
        )
        ok = await webhook.notify(cfg, _event(), "hi")
    assert ok is False


@pytest.mark.asyncio
async def test_missing_received_at_serialized_as_null():
    cfg = WebhookConfig(enabled=True, url="https://example.com/hook")
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").mock(side_effect=_handler)
        await webhook.notify(cfg, _event(received_at=0.0), "hi")

    assert captured["received_at"] is None


# --- Bark-native format (format = "bark") ---------------------------------


@pytest.mark.asyncio
async def test_bark_format_builds_native_payload():
    cfg = WebhookConfig(
        enabled=True, url="https://api.day.app/devkey", format="bark",
    )
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    with respx.mock(base_url="https://api.day.app") as router:
        router.post("/devkey").mock(side_effect=_handler)
        ok = await webhook.notify(
            cfg,
            _event(cwd="/Users/me/myself/jarvis-cli"),
            "Sir, the build is done.",
        )

    assert ok is True
    assert captured == {
        "title": "Jarvis · jarvis-cli",
        "body": "Sir, the build is done.",
        "group": "jarvis-cli",
        "level": "active",  # idle_prompt is not attention-needed
    }


@pytest.mark.asyncio
async def test_bark_attention_types_are_time_sensitive():
    cfg = WebhookConfig(
        enabled=True, url="https://api.day.app/devkey", format="bark",
    )
    levels: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        levels.append(json.loads(request.content)["level"])
        return httpx.Response(200)

    with respx.mock(base_url="https://api.day.app") as router:
        router.post("/devkey").mock(side_effect=_handler)
        for ntype in ("permission_prompt", "tool_failure", "rate_limited"):
            await webhook.notify(cfg, _event(notification_type=ntype), "hi")

    assert levels == ["timeSensitive"] * 3


@pytest.mark.asyncio
async def test_bark_missing_cwd_falls_back_to_jarvis():
    cfg = WebhookConfig(
        enabled=True, url="https://api.day.app/devkey", format="bark",
    )
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    with respx.mock(base_url="https://api.day.app") as router:
        router.post("/devkey").mock(side_effect=_handler)
        await webhook.notify(cfg, _event(cwd=None), "hi")

    assert captured["title"] == "Jarvis · jarvis"
    assert captured["group"] == "jarvis"


@pytest.mark.asyncio
async def test_default_format_stays_generic():
    """Back-compat: an untouched WebhookConfig must keep the flat payload."""
    cfg = WebhookConfig(enabled=True, url="https://example.com/hook")
    assert cfg.format == "generic"
    captured: dict = {}

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200)

    with respx.mock(base_url="https://example.com") as router:
        router.post("/hook").mock(side_effect=_handler)
        await webhook.notify(cfg, _event(), "hi")

    assert "text" in captured and "title" not in captured


@pytest.mark.asyncio
async def test_ntfy_format_posts_text_body_with_headers():
    """format="ntfy" sends the spoken line as the raw body with Title/
    Priority headers — one app (ntfy) covers notify-only pushes here and
    actionable ones in notify/remote.py."""
    cfg = WebhookConfig(
        enabled=True, url="https://ntfy.sh/some-topic", format="ntfy",
    )
    with respx.mock(base_url="https://ntfy.sh") as router:
        route = router.post("/some-topic").respond(200)
        ok = await webhook.notify(
            cfg,
            _event(notification_type="tool_failure", cwd="/repo/jarvis-cli"),
            "Sir, the tests have failed.",
        )

    assert ok is True
    request = route.calls[0].request
    assert request.content.decode() == "Sir, the tests have failed."
    assert request.headers["Title"] == "Jarvis - jarvis-cli"
    assert request.headers["Priority"] == "high"  # attention type
    assert request.headers["Tags"] == "robot"
