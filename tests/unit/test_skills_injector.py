from pathlib import Path

from jarvis_cli.skills.catalog import SkillRecord
from jarvis_cli.skills.injector import (
    InjectionPolicy,
    build_clarify,
    build_injection,
    gate_matches,
)
from jarvis_cli.skills.retriever import Match


def _rec(name: str, tmp: Path, body: str = "BODY-CONTENT") -> SkillRecord:
    p = tmp / "skills" / name / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\nname: {name}\ndescription: {name} desc\n---\n{body}\n",
                 encoding="utf-8")
    return SkillRecord(name=name, description=f"{name} desc", source_tool="claude",
                       path=str(p))


def test_strong_match_injects_body(tmp_path):
    m = [Match(_rec("alpha", tmp_path), 0.9, whole_word=True)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "body"
    assert "BODY-CONTENT" in res.text
    assert res.injected_keys == ["alpha"]


def test_medium_match_injects_menu(tmp_path):
    m = [Match(_rec("beta", tmp_path), 0.35, whole_word=True)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "menu"
    assert "`beta`" in res.text
    assert res.injected_keys == []


def test_weak_match_injects_nothing(tmp_path):
    m = [Match(_rec("gamma", tmp_path), 0.1, whole_word=True)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "none"
    assert res.text is None


def test_already_injected_strong_is_skipped(tmp_path):
    m = [Match(_rec("delta", tmp_path), 0.9, whole_word=True)]
    res = build_injection(m, policy=InjectionPolicy(), already_injected={"delta"})
    assert res.mode == "none"
    assert res.text is None


def test_max_skills_caps_bodies(tmp_path):
    m = [
        Match(_rec("s1", tmp_path), 0.9, whole_word=True),
        Match(_rec("s2", tmp_path), 0.85, whole_word=True),
        Match(_rec("s3", tmp_path), 0.8, whole_word=True),
    ]
    res = build_injection(m, policy=InjectionPolicy(max_skills=2))
    assert res.mode == "body"
    assert len(res.injected_keys) == 2


def test_total_char_budget_truncates_set(tmp_path):
    big = "X" * 5000
    m = [
        Match(_rec("b1", tmp_path, body=big), 0.9, whole_word=True),
        Match(_rec("b2", tmp_path, body=big), 0.9, whole_word=True),
    ]
    res = build_injection(
        m, policy=InjectionPolicy(max_skills=2, total_char_budget=6000)
    )
    # second body would blow the budget, so only the first is kept
    assert len(res.injected_keys) == 1


def test_pure_semantic_without_keywords_filtered(tmp_path):
    # hybrid=0.35, cosine=0.35, no whole-word hit -> cosine < 0.50 -> gate rejects
    m = [Match(_rec("epsilon", tmp_path), 0.35, cosine=0.35)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "none"
    assert res.text is None


def test_strong_cosine_alone_passes_gate(tmp_path):
    # hybrid=0.55, cosine=0.55 -> no lexical hit, but cosine >= 0.50 -> passes
    m = [Match(_rec("zeta", tmp_path), 0.55, cosine=0.55)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "body"
    assert "BODY-CONTENT" in res.text


# -- vague-query guard: short/generic prompts must not command-inject a body
#    unless a whole-word hit names the target --

def test_vague_query_demotes_body_without_named_target(tmp_path):
    # "更新一下" is vague; the match passes the gate on cosine alone but has no
    # whole-word hit, so it must fall back to a menu instead of a body.
    m = [Match(_rec("update-deps", tmp_path), 0.9, cosine=0.55, whole_word=False)]
    res = build_injection(m, policy=InjectionPolicy(), query="更新一下")
    assert res.mode == "menu"
    assert res.injected_keys == []


def test_vague_query_keeps_body_when_target_named(tmp_path):
    # "部署 vercel" is short but names a target (whole-word hit) -> body allowed.
    m = [Match(_rec("deploy-to-vercel", tmp_path), 0.9, whole_word=True)]
    res = build_injection(m, policy=InjectionPolicy(), query="部署 vercel")
    assert res.mode == "body"
    assert "BODY-CONTENT" in res.text


def test_nonvague_query_keeps_body_without_named_target(tmp_path):
    # A spelled-out request is not vague, so a strong cosine-only match still
    # body-injects.
    m = [Match(_rec("update-deps", tmp_path), 0.9, cosine=0.55, whole_word=False)]
    res = build_injection(
        m, policy=InjectionPolicy(),
        query="请帮我把这个项目的所有依赖都升级到最新版本",
    )
    assert res.mode == "body"


# -- build_clarify: ambiguous candidates surfaced as a question, not a command --

def test_build_clarify_lists_candidates(tmp_path):
    m = [
        Match(_rec("update-deps", tmp_path), 0.5),
        Match(_rec("update-config", tmp_path), 0.45),
    ]
    res = build_clarify(m)
    assert res.mode == "clarify"
    assert "`update-deps`" in res.text
    assert "`update-config`" in res.text
    # the note must tell the model to ask first, not act
    assert "clarify" in res.text.lower()
    assert res.injected_keys == []


def test_build_clarify_empty_is_none(tmp_path):
    res = build_clarify([])
    assert res.mode == "none"
    assert res.text is None


# -- gate_matches: the standalone gate shared with SkillService.gate / MCP --

def test_gate_matches_keeps_wholeword_and_strong_cosine(tmp_path):
    wholeword = Match(_rec("a", tmp_path), 0.30, cosine=0.05, whole_word=True)
    cosine_solo = Match(_rec("b", tmp_path), 0.55, cosine=0.55)  # cosine >= .50
    bigram_only = Match(_rec("c", tmp_path), 0.30, cosine=0.05)  # boost but no whole word
    semantic = Match(_rec("d", tmp_path), 0.35, cosine=0.35)  # pure semantic
    weak = Match(_rec("e", tmp_path), 0.10, cosine=0.02)  # below med
    kept = gate_matches(
        [wholeword, cosine_solo, bigram_only, semantic, weak], med_threshold=0.28
    )
    assert [m.record.name for m in kept] == ["a", "b"]


def test_gate_matches_empty_when_nothing_passes(tmp_path):
    semantic = Match(_rec("c", tmp_path), 0.35, cosine=0.35)
    assert gate_matches([semantic], med_threshold=0.28) == []
