"""Test MCP service: gate logic (McpService.query) and injection building.

Match(record, hybrid_score, cosine) -- the lexical boost is the delta
between hybrid and cosine.  The gate in McpService.query requires lexical
signal (boost > 0) OR cosine >= 0.50; _build_injection receives
pre-filtered candidates from the daemon.
"""
from jarvis_cli.mcp.registry import McpServerRecord
from jarvis_cli.mcp.service import (
    _build_injection,
    _has_lexical_signal,
    _COSINE_SOLO_FLOOR,
)
from jarvis_cli.retrieval.retriever import Match


def _rec(name: str, desc: str = "test server", **connect_kw) -> McpServerRecord:
    return McpServerRecord(
        name=name, description=desc,
        connect={"type": "http", "url": "http://localhost", **connect_kw},
    )


def _passes_gate(m: Match, med: float = 0.22) -> bool:
    return m.score >= med and (
        _has_lexical_signal(m) or m.cosine >= _COSINE_SOLO_FLOOR
    )


# -- Gate logic tests --

def test_gate_passes_with_lexical_signal():
    m = Match(_rec("memex"), 0.30, cosine=0.08)
    assert _passes_gate(m)


def test_gate_passes_with_strong_cosine():
    m = Match(_rec("memex"), 0.55, cosine=0.55)
    assert _passes_gate(m)


def test_gate_rejects_pure_semantic():
    m = Match(_rec("thinking"), 0.25, cosine=0.25)
    assert not _passes_gate(m)


def test_gate_rejects_weak():
    m = Match(_rec("irrelevant"), 0.1, cosine=0.1)
    assert not _passes_gate(m)


# -- Injection building tests (pre-filtered candidates) --

def test_strong_match_injects_connect():
    matches = [Match(_rec("memex", "session history search"), 0.9, cosine=0.68)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert ctx is not None
    assert "memex" in ctx
    assert "add_server" in ctx


def test_medium_match_injects_menu():
    matches = [Match(_rec("ddg", "web search"), 0.30, cosine=0.08)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert ctx is not None
    assert "`ddg`" in ctx


def test_empty_candidates_returns_none():
    ctx = _build_injection([], high_threshold=0.35, med_threshold=0.22)
    assert ctx is None


def test_multiple_strong_matches():
    matches = [
        Match(_rec("memex"), 0.9, cosine=0.68),
        Match(_rec("playwright"), 0.85, cosine=0.63),
    ]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert ctx is not None
    assert "memex" in ctx
    assert "playwright" in ctx


def test_connect_params_in_snippet():
    rec = McpServerRecord(
        name="custom",
        description="custom server",
        connect={"type": "stdio", "command": "npx", "args": ["my-mcp"]},
    )
    matches = [Match(rec, 0.9, cosine=0.68)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert '"command": "npx"' in ctx
    assert '"args": ["my-mcp"]' in ctx
