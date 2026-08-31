import pytest

from jarvis.config import Config
from jarvis.phrase.providers.base import PhraseProvider
from jarvis.phrase.router import PhraseRouter
from jarvis.types import Event


class _CapturingStub(PhraseProvider):
    name = "cap"

    def __init__(self, output: str = "<ok>") -> None:
        self.output = output
        self.last_messages: list[dict[str, str]] | None = None

    async def generate(self, messages):
        self.last_messages = messages
        return self.output


@pytest.mark.asyncio
async def test_router_passes_extracted_summary_into_user_blob():
    stub = _CapturingStub()
    router = PhraseRouter(stub, None, Config())
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "rm -rf /tmp/xyz"},
    )
    await router.phrase(ev, lang="en")
    assert stub.last_messages is not None
    user_blob = stub.last_messages[-1]["content"]
    assert "rm -rf /tmp/xyz" in user_blob
    # Raw key "tool_input" must NOT be in the prompt
    assert "tool_input" not in user_blob


@pytest.mark.asyncio
async def test_router_redacts_home_path_when_enabled(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/jobin")
    stub = _CapturingStub()
    cfg = Config()
    cfg.behavior.privacy.cloud_redaction = True
    router = PhraseRouter(stub, None, cfg)
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "rm -rf /Users/jobin/tmp/x"},
    )
    await router.phrase(ev, lang="en")
    blob = stub.last_messages[-1]["content"]
    assert "rm -rf ~/tmp/x" in blob
    assert "/Users/jobin" not in blob


@pytest.mark.asyncio
async def test_router_skips_redaction_when_disabled(monkeypatch):
    monkeypatch.setenv("HOME", "/Users/jobin")
    stub = _CapturingStub()
    cfg = Config()
    cfg.behavior.privacy.cloud_redaction = False
    router = PhraseRouter(stub, None, cfg)
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "ls /Users/jobin/tmp"},
    )
    await router.phrase(ev, lang="en")
    blob = stub.last_messages[-1]["content"]
    assert "/Users/jobin/tmp" in blob


@pytest.mark.asyncio
async def test_router_does_not_post_truncate_long_llm_output():
    long_output = "Sir, " + "x" * 200  # ~205 chars, well beyond old 30 cap
    stub = _CapturingStub(output=long_output)
    router = PhraseRouter(stub, None, Config())
    out = await router.phrase(
        Event(notification_type="idle_prompt", tool_name=None),
        lang="en",
    )
    assert out == long_output  # unchanged — no router-side truncation


@pytest.mark.asyncio
async def test_router_empty_tool_input_passes_empty_summary():
    stub = _CapturingStub()
    router = PhraseRouter(stub, None, Config())
    ev = Event(notification_type="idle_prompt", tool_name=None, tool_input={})
    await router.phrase(ev, lang="en")
    blob = stub.last_messages[-1]["content"]
    assert '"summary": ""' in blob or '"summary":""' in blob
