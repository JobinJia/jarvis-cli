"""task_complete variety: same flavor + avoid mechanics as idle_prompt.

Before this existed the fixed few-shot answer made qwen3:8b say
"All done, sir." for every completion — 86 times in five days, which the
user heard as a chant. The flavor hint and previous-line avoid are what
actually move a small model (examples beat clauses, 2026-07-09 probe).
"""
import json

import pytest

from jarvis_cli.config import Config
from jarvis_cli.phrase.prompt import _COMPLETE_FLAVORS, build_messages
from jarvis_cli.phrase.providers.base import PhraseProvider
from jarvis_cli.phrase.router import PhraseRouter
from jarvis_cli.types import Event


def _complete_event() -> Event:
    return Event(notification_type="task_complete", tool_name=None, tool_input={})


def _build(
    event: Event, avoid: str | list[str] | None = None,
) -> list[dict[str, str]]:
    return build_messages(
        event, "en", "", target_chars=70, hard_cap=120, avoid=avoid,
    )


def test_complete_blob_carries_flavor_from_hint_list():
    blob = json.loads(_build(_complete_event())[-1]["content"])
    assert blob["flavor"] in _COMPLETE_FLAVORS


def test_complete_blob_carries_avoid_when_given():
    blob = json.loads(
        _build(_complete_event(), avoid=["All done, sir."])[-1]["content"]
    )
    assert blob["avoid"] == ["All done, sir."]


def test_complete_blob_carries_the_whole_avoid_window():
    # The endings collapsed onto one tail while the openings varied, so the
    # model must see several recent lines at once, not just the last one.
    recent = ["All done, sir.", "The letter is posted - all is settled."]
    blob = json.loads(_build(_complete_event(), avoid=recent)[-1]["content"])
    assert blob["avoid"] == recent


def test_complete_blob_wraps_a_bare_avoid_string():
    blob = json.loads(
        _build(_complete_event(), avoid="All done, sir.")[-1]["content"]
    )
    assert blob["avoid"] == ["All done, sir."]


def test_complete_blob_omits_avoid_when_none():
    blob = json.loads(_build(_complete_event())[-1]["content"])
    assert "avoid" not in blob


def test_complete_system_prompt_demands_fresh_phrasing():
    sys = _build(_complete_event())[0]["content"]
    assert "flavor" in sys
    assert "avoid" in sys


class _CapturingStub(PhraseProvider):
    name = "cap"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[dict[str, str]]] = []

    async def generate(self, messages):
        self.calls.append(messages)
        return self.outputs[len(self.calls) - 1]


@pytest.mark.asyncio
async def test_router_threads_previous_complete_line_as_avoid():
    stub = _CapturingStub(["Dinner is served, sir.", "The lap is won, sir."])
    router = PhraseRouter(stub, None, Config())

    first = await router.phrase(_complete_event(), lang="en")
    assert first == "Dinner is served, sir."
    first_blob = json.loads(stub.calls[0][-1]["content"])
    assert "avoid" not in first_blob

    await router.phrase(_complete_event(), lang="en")
    second_blob = json.loads(stub.calls[1][-1]["content"])
    assert second_blob["avoid"] == ["Dinner is served, sir."]


@pytest.mark.asyncio
async def test_router_threads_a_window_of_recent_complete_lines():
    spoken = [
        "One, sir.", "Two, sir.", "Three, sir.",
        "Four, sir.", "Five, sir.", "Six, sir.",
    ]
    stub = _CapturingStub(spoken)
    router = PhraseRouter(stub, None, Config())
    for _ in spoken:
        await router.phrase(_complete_event(), lang="en")

    # The 6th request sees the 5 lines before it, oldest first.
    assert json.loads(stub.calls[5][-1]["content"])["avoid"] == spoken[:5]
    # The window is bounded: the oldest line has dropped off.
    router._remember_line(_complete_event(), "Seven, sir.")
    assert router._avoid_for(_complete_event()) == spoken[2:] + ["Seven, sir."]


@pytest.mark.asyncio
async def test_repeated_line_does_not_flush_the_avoid_window():
    # A model stuck on one phrase must not evict its own history and forget
    # everything else it just said.
    stub = _CapturingStub(["One, sir.", "Two, sir.", "One, sir."])
    router = PhraseRouter(stub, None, Config())
    for _ in range(3):
        await router.phrase(_complete_event(), lang="en")

    assert router._avoid_for(_complete_event()) == ["Two, sir.", "One, sir."]


@pytest.mark.asyncio
async def test_idle_and_complete_avoid_lines_do_not_cross():
    stub = _CapturingStub(["Your move, sir.", "Dinner is served, sir."])
    router = PhraseRouter(stub, None, Config())
    idle = Event(notification_type="idle_prompt", tool_name=None, tool_input={})
    await router.phrase(idle, lang="en")

    await router.phrase(_complete_event(), lang="en")
    blob = json.loads(stub.calls[1][-1]["content"])
    # The completion request must not inherit the idle line as avoid.
    assert "avoid" not in blob
