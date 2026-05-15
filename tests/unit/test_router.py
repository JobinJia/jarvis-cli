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
