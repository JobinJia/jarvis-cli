"""idle_prompt variety: every request carries a random flavor hint and the
previous line to avoid, so the LLM improvises instead of parroting."""
import json

import pytest

from jarvis_cli.config import Config
from jarvis_cli.phrase.prompt import _IDLE_FLAVORS, build_messages
from jarvis_cli.phrase.providers.base import PhraseProvider
from jarvis_cli.phrase.router import PhraseRouter
from jarvis_cli.types import Event


def _idle_event() -> Event:
    return Event(notification_type="idle_prompt", tool_name=None, tool_input={})


def _build(event: Event, avoid: str | None = None) -> list[dict[str, str]]:
    return build_messages(
        event, "en", "", target_chars=70, hard_cap=120, avoid=avoid,
    )


def test_idle_blob_carries_flavor_from_hint_list():
    blob = json.loads(_build(_idle_event())[-1]["content"])
    assert blob["flavor"] in _IDLE_FLAVORS


def test_idle_blob_carries_avoid_when_given():
    blob = json.loads(_build(_idle_event(), avoid="Sir, old line.")[-1]["content"])
    assert blob["avoid"] == "Sir, old line."


def test_idle_blob_omits_avoid_when_none():
    blob = json.loads(_build(_idle_event())[-1]["content"])
    assert "avoid" not in blob


def test_idle_system_prompt_demands_fresh_phrasing():
    sys = _build(_idle_event())[0]["content"]
    assert "flavor" in sys
    assert "avoid" in sys


def test_non_idle_blob_gains_no_flavor_or_avoid():
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "ls"},
    )
    messages = build_messages(
        ev, "en", "ls", target_chars=70, hard_cap=120, avoid="should be ignored",
    )
    blob = json.loads(messages[-1]["content"])
    assert "flavor" not in blob
    assert "avoid" not in blob
    sys = messages[0]["content"]
    assert "flavor" not in sys


class _CapturingStub(PhraseProvider):
    name = "cap"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages):
        self.calls.append(messages)
        return self.outputs[len(self.calls) - 1]


@pytest.mark.asyncio
async def test_router_threads_previous_idle_line_as_avoid():
    stub = _CapturingStub(["Your move, sir.", "All quiet, sir."])
    router = PhraseRouter(stub, None, Config())

    first = await router.phrase(_idle_event(), lang="en")
    assert first == "Your move, sir."
    first_blob = json.loads(stub.calls[0][-1]["content"])
    assert "avoid" not in first_blob

    await router.phrase(_idle_event(), lang="en")
    second_blob = json.loads(stub.calls[1][-1]["content"])
    assert second_blob["avoid"] == "Your move, sir."


@pytest.mark.asyncio
async def test_router_does_not_thread_avoid_into_non_idle_events():
    stub = _CapturingStub(["Your move, sir.", "Sir, he wants Bash."])
    router = PhraseRouter(stub, None, Config())
    await router.phrase(_idle_event(), lang="en")

    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "ls"},
    )
    await router.phrase(ev, lang="en")
    blob = json.loads(stub.calls[1][-1]["content"])
    assert "avoid" not in blob
