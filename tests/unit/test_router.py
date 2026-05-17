from typing import Literal

import pytest

from jarvis_cc.config import Config
from jarvis_cc.phrase.providers.base import PhraseProvider
from jarvis_cc.phrase.router import PhraseRouter
from jarvis_cc.types import Event


class _Stub(PhraseProvider):
    def __init__(self, name: str, mode: Literal["ok", "fail"]) -> None:
        self.name = name
        self.mode = mode
        self.calls = 0
        self.last_messages: list[dict[str, str]] | None = None

    async def generate(self, messages):
        self.calls += 1
        self.last_messages = messages
        if self.mode == "fail":
            raise RuntimeError(f"{self.name} down")
        return f"<{self.name}>"


def _ev() -> Event:
    return Event(notification_type="idle_prompt", tool_name=None)


@pytest.mark.asyncio
async def test_router_returns_primary_when_healthy():
    primary, fallback = _Stub("p", "ok"), _Stub("f", "ok")
    router = PhraseRouter(primary, fallback, Config())
    out = await router.phrase(_ev(), lang="zh")
    assert out == "<p>"
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_router_falls_back_when_primary_fails():
    primary, fallback = _Stub("p", "fail"), _Stub("f", "ok")
    router = PhraseRouter(primary, fallback, Config())
    out = await router.phrase(_ev(), lang="zh")
    assert out == "<f>"
    assert primary.calls == 1
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_router_returns_template_when_both_fail():
    primary, fallback = _Stub("p", "fail"), _Stub("f", "fail")
    router = PhraseRouter(primary, fallback, Config())
    out = await router.phrase(_ev(), lang="zh")
    assert "先生" in out


@pytest.mark.asyncio
async def test_router_template_when_no_fallback():
    primary = _Stub("p", "fail")
    router = PhraseRouter(primary, None, Config())
    out = await router.phrase(_ev(), lang="en")
    assert "Sir" in out


@pytest.mark.asyncio
async def test_router_invokes_callback_when_primary_falls_back():
    """When the primary phrase provider fails (ollama down) and the
    fallback (deepseek) succeeds, the router must surface that transition
    so callers can notify the user — otherwise local-first deployments
    silently start burning cloud credits."""
    primary, fallback = _Stub("p", "fail"), _Stub("f", "ok")
    fired: list[str] = []

    async def on_fallback(primary_name: str) -> None:
        fired.append(primary_name)

    router = PhraseRouter(
        primary, fallback, Config(),
        on_primary_fallback=on_fallback,
    )
    out = await router.phrase(_ev(), lang="en")
    assert out == "<f>"
    assert fired == ["p"], f"expected one callback with primary name, got {fired!r}"


@pytest.mark.asyncio
async def test_router_does_not_invoke_callback_when_primary_healthy():
    """Healthy primary path must NOT trigger the fallback notification."""
    primary, fallback = _Stub("p", "ok"), _Stub("f", "ok")
    fired: list[str] = []

    async def on_fallback(primary_name: str) -> None:
        fired.append(primary_name)

    router = PhraseRouter(
        primary, fallback, Config(),
        on_primary_fallback=on_fallback,
    )
    await router.phrase(_ev(), lang="en")
    assert fired == []


@pytest.mark.asyncio
async def test_router_does_not_invoke_callback_when_both_fail():
    """If both fail and we end up at the template, there's nothing useful
    to alert about — we already produced a degraded result; spamming a
    voice alert would just stack on top of the template output."""
    primary, fallback = _Stub("p", "fail"), _Stub("f", "fail")
    fired: list[str] = []

    async def on_fallback(primary_name: str) -> None:
        fired.append(primary_name)

    router = PhraseRouter(
        primary, fallback, Config(),
        on_primary_fallback=on_fallback,
    )
    await router.phrase(_ev(), lang="en")
    assert fired == []
