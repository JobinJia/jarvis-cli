"""Test the shared LLM intent verifier (skills + MCP).

The verifier classifies gate-passed candidates into confirmed / none / unclear.
Failure modes (Ollama down, unparseable reply) must resolve to ``unclear`` — we
ask the user rather than fabricate a confirm or command-inject everything the
gate let through.
"""
import httpx
import respx

from jarvis_cli.config import OllamaConfig
from jarvis_cli.retrieval.retriever import Match
from jarvis_cli.retrieval.verifier import (
    CONFIRMED,
    NONE,
    UNCLEAR,
    verify_candidates,
)


class _Rec:
    def __init__(self, name: str, description: str = "desc") -> None:
        self.name = name
        self.description = description


def _match(name: str) -> Match:
    return Match(_Rec(name), 0.5, cosine=0.4)


def _candidates() -> list[Match]:
    return [_match("update-deps"), _match("git-commit"), _match("deep-research")]


async def test_confirmed_returns_only_llm_selected_subset():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200, json={"message": {"content": "2"}},
        )
        out = await verify_candidates("提交代码", _candidates(), cfg, noun="skill")
    assert out.status == CONFIRMED
    assert [m.record.name for m in out.matches] == ["git-commit"]


async def test_none_reply_drops_all():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200, json={"message": {"content": "none"}},
        )
        out = await verify_candidates("更新项目库", _candidates(), cfg, noun="skill")
    assert out.status == NONE
    assert out.matches == []


async def test_unclear_reply_holds_back():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200, json={"message": {"content": "unclear"}},
        )
        out = await verify_candidates("更新一下", _candidates(), cfg, noun="skill")
    assert out.status == UNCLEAR
    assert out.matches == []


async def test_multiple_indices_preserve_candidate_order():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200, json={"message": {"content": "3, 1"}},
        )
        out = await verify_candidates("x", _candidates(), cfg, noun="skill")
    assert out.status == CONFIRMED
    assert [m.record.name for m in out.matches] == ["update-deps", "deep-research"]


async def test_unavailable_llm_is_unclear():
    # Nothing listening -> connection error -> unclear (ask the user), NOT a
    # fabricated confirm of every candidate.
    cfg = OllamaConfig(base_url="http://127.0.0.1:1")
    out = await verify_candidates("x", _candidates(), cfg, noun="skill", timeout_s=0.2)
    assert out.status == UNCLEAR
    assert out.matches == []


async def test_unparseable_reply_is_unclear():
    # No "none", no "unclear", no digits -> we couldn't read the verdict, so
    # hold back instead of passing every candidate through.
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").respond(
            200, json={"message": {"content": "yes please"}},
        )
        out = await verify_candidates("x", _candidates(), cfg, noun="skill")
    assert out.status == UNCLEAR
    assert out.matches == []


async def test_http_error_is_unclear():
    cfg = OllamaConfig()
    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/api/chat").mock(side_effect=httpx.ConnectError("boom"))
        out = await verify_candidates("x", _candidates(), cfg, noun="tool server")
    assert out.status == UNCLEAR
    assert out.matches == []


async def test_empty_candidates_short_circuits():
    cfg = OllamaConfig()
    out = await verify_candidates("x", [], cfg, noun="skill")
    assert out.status == NONE
    assert out.matches == []
