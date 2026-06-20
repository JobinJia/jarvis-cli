"""Test MCP service injection logic (no embedding model needed).

Match(record, hybrid_score, cosine) — the lexical boost is the delta
between hybrid and cosine.  The gate requires lexical signal (boost > 0)
OR cosine >= 0.50, so test fixtures set cosine explicitly.
"""
from jarvis_cli.mcp.registry import McpServerRecord
from jarvis_cli.mcp.service import _build_injection
from jarvis_cli.retrieval.retriever import Match


def _rec(name: str, desc: str = "test server", **connect_kw) -> McpServerRecord:
    return McpServerRecord(
        name=name, description=desc,
        connect={"type": "http", "url": "http://localhost", **connect_kw},
    )


def test_strong_match_with_lexical_signal():
    # hybrid 0.9, cosine 0.68 → lexical boost 0.22 → passes gate
    matches = [Match(_rec("memex", "session history search"), 0.9, cosine=0.68)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert ctx is not None
    assert "memex" in ctx
    assert "add_server" in ctx


def test_strong_cosine_alone_passes_gate():
    # hybrid 0.55, cosine 0.55 → no lexical boost, but cosine >= 0.50
    matches = [Match(_rec("memex", "session history"), 0.55, cosine=0.55)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert ctx is not None
    assert "memex" in ctx


def test_medium_match_with_lexical_signal():
    # hybrid 0.30, cosine 0.08 → lexical boost 0.22 → passes gate
    matches = [Match(_rec("ddg", "web search"), 0.30, cosine=0.08)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert ctx is not None
    assert "`ddg`" in ctx


def test_pure_semantic_without_keywords_filtered():
    # hybrid 0.25, cosine 0.25 → no lexical boost, cosine < 0.50 → gate rejects
    matches = [Match(_rec("thinking", "reasoning"), 0.25, cosine=0.25)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
    assert ctx is None


def test_weak_match_injects_nothing():
    matches = [Match(_rec("irrelevant"), 0.1, cosine=0.1)]
    ctx = _build_injection(matches, high_threshold=0.35, med_threshold=0.22)
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
