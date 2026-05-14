import os

import httpx
import pytest
import respx

from jarvis_cc.config import DeepSeekConfig, OllamaConfig
from jarvis_cc.phrase.providers.deepseek import DeepSeekProvider
from jarvis_cc.phrase.providers.ollama import OllamaProvider
from jarvis_cc.types import Event


def _ev() -> Event:
    return Event(notification_type="permission_prompt", tool_name="Bash")


@pytest.mark.asyncio
async def test_deepseek_returns_assistant_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    cfg = DeepSeekConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "先生，Claude 请求许可。"}}
                ]
            },
        )
        p = DeepSeekProvider(cfg)
        out = await p.generate(_ev(), lang="zh", max_chars=30)
    assert out == "先生，Claude 请求许可。"


@pytest.mark.asyncio
async def test_deepseek_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = DeepSeekProvider(DeepSeekConfig())
    with pytest.raises(RuntimeError):
        await p.generate(_ev(), lang="zh", max_chars=30)


@pytest.mark.asyncio
async def test_deepseek_raises_on_http_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    cfg = DeepSeekConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/v1/chat/completions").respond(500)
        p = DeepSeekProvider(cfg)
        with pytest.raises(httpx.HTTPStatusError):
            await p.generate(_ev(), lang="zh", max_chars=30)


@pytest.mark.asyncio
async def test_ollama_returns_assistant_text():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200,
            json={"message": {"role": "assistant", "content": "先生，请过目。"}},
        )
        p = OllamaProvider(cfg)
        out = await p.generate(_ev(), lang="zh", max_chars=30)
    assert out == "先生，请过目。"


@pytest.mark.asyncio
async def test_ollama_raises_on_connection_error():
    cfg = OllamaConfig(base_url="http://127.0.0.1:1")  # nothing listens here
    p = OllamaProvider(cfg)
    with pytest.raises(httpx.HTTPError):
        await p.generate(_ev(), lang="zh", max_chars=30)
