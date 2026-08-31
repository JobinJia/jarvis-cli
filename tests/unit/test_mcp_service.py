"""Test MCP service: gate logic and injection building.

Match(record, hybrid_score, cosine, whole_word) -- the gate (shared
``gate_matches``) requires a whole-word lexical hit OR cosine >= 0.50; a
bigram-only overlap on a common word no longer qualifies. ``_build_injection``
receives pre-filtered, intent-confirmed candidates from the daemon.
"""
from jarvis.mcp.registry import McpServerRecord
from jarvis.mcp.service import _build_clarify, _build_injection
from jarvis.retrieval.retriever import Match, gate_matches


def _rec(name: str, desc: str = "test server", **connect_kw) -> McpServerRecord:
    return McpServerRecord(
        name=name, description=desc,
        connect={"type": "http", "url": "http://localhost", **connect_kw},
    )


def _passes_gate(m: Match, med: float = 0.22) -> bool:
    return bool(gate_matches([m], med_threshold=med))


# -- Gate logic tests --

def test_gate_passes_with_wholeword_signal():
    m = Match(_rec("memex"), 0.30, cosine=0.08, whole_word=True)
    assert _passes_gate(m)


def test_gate_passes_with_strong_cosine():
    m = Match(_rec("memex"), 0.55, cosine=0.55)
    assert _passes_gate(m)


def test_gate_rejects_bigram_only_overlap():
    # boost lifted the hybrid score, but no whole-word hit and cosine < 0.50:
    # a common-word bigram overlap must not pass.
    m = Match(_rec("memex"), 0.30, cosine=0.08, whole_word=False)
    assert not _passes_gate(m)


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


# -- Vague-query guard: a thin prompt with no named target stays off the
#    connect tier even when the candidate is strong. --

def test_vague_query_demotes_connect_to_menu():
    matches = [Match(_rec("memex"), 0.9, cosine=0.55, whole_word=False)]
    ctx = _build_injection(
        matches, high_threshold=0.35, med_threshold=0.22, query="看一下",
    )
    assert ctx is not None
    assert "**Connect:**" not in ctx  # not the connect/body tier
    assert "`memex`" in ctx           # menu tier


def test_vague_query_keeps_connect_when_target_named():
    matches = [Match(_rec("memex"), 0.9, cosine=0.55, whole_word=True)]
    ctx = _build_injection(
        matches, high_threshold=0.35, med_threshold=0.22, query="memex 看一下",
    )
    assert "**Connect:**" in ctx


# -- _build_clarify: ambiguous candidates surfaced as a question --

def test_build_clarify_lists_candidates():
    matches = [Match(_rec("memex", "session history"), 0.5, cosine=0.3)]
    ctx = _build_clarify(matches)
    assert ctx is not None
    assert "`memex`" in ctx
    assert "add_server" not in ctx
    assert "clarify" in ctx.lower()


def test_build_clarify_empty_returns_none():
    assert _build_clarify([]) is None
