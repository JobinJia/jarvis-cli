import pytest
import respx

from jarvis.config import AnthropicConfig, OpenAIConfig
from jarvis.phrase.providers.anthropic import AnthropicProvider
from jarvis.phrase.providers.openai import OpenAIProvider

_MESSAGES = [
    {"role": "system", "content": "you are jarvis"},
    {"role": "user", "content": '{"notification_type":"permission_prompt","tool_name":"Bash","summary":"rm /tmp/x"}'},
]


@pytest.mark.asyncio
async def test_anthropic_returns_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = AnthropicConfig()
    with respx.mock(base_url="https://api.anthropic.com") as router:
        router.post("/v1/messages").respond(
            200,
            json={
                "id": "m_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Sir, your shell awaits."}],
                "model": cfg.model,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        p = AnthropicProvider(cfg)
        out = await p.generate(_MESSAGES)
    assert out == "Sir, your shell awaits."


@pytest.mark.asyncio
async def test_openai_returns_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    cfg = OpenAIConfig()
    with respx.mock(base_url="https://api.openai.com") as router:
        router.post("/v1/chat/completions").respond(
            200,
            json={
                "choices": [{"message": {"content": "Sir, at your discretion."}}],
            },
        )
        p = OpenAIProvider(cfg)
        out = await p.generate(_MESSAGES)
    assert out == "Sir, at your discretion."
