from pathlib import Path

from jarvis_cli.skills.catalog import SkillRecord
from jarvis_cli.skills.injector import InjectionPolicy, build_injection
from jarvis_cli.skills.retriever import Match


def _rec(name: str, tmp: Path, body: str = "BODY-CONTENT") -> SkillRecord:
    p = tmp / "skills" / name / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\nname: {name}\ndescription: {name} desc\n---\n{body}\n",
                 encoding="utf-8")
    return SkillRecord(name=name, description=f"{name} desc", source_tool="claude",
                       path=str(p))


def test_strong_match_injects_body(tmp_path):
    m = [Match(_rec("alpha", tmp_path), 0.9)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "body"
    assert "BODY-CONTENT" in res.text
    assert res.injected_keys == ["alpha"]


def test_medium_match_injects_menu(tmp_path):
    m = [Match(_rec("beta", tmp_path), 0.35)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "menu"
    assert "`beta`" in res.text
    assert res.injected_keys == []


def test_weak_match_injects_nothing(tmp_path):
    m = [Match(_rec("gamma", tmp_path), 0.1)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "none"
    assert res.text is None


def test_already_injected_strong_is_skipped(tmp_path):
    m = [Match(_rec("delta", tmp_path), 0.9)]
    res = build_injection(m, policy=InjectionPolicy(), already_injected={"delta"})
    assert res.mode == "none"
    assert res.text is None


def test_max_skills_caps_bodies(tmp_path):
    m = [
        Match(_rec("s1", tmp_path), 0.9),
        Match(_rec("s2", tmp_path), 0.85),
        Match(_rec("s3", tmp_path), 0.8),
    ]
    res = build_injection(m, policy=InjectionPolicy(max_skills=2))
    assert res.mode == "body"
    assert len(res.injected_keys) == 2


def test_total_char_budget_truncates_set(tmp_path):
    big = "X" * 5000
    m = [
        Match(_rec("b1", tmp_path, body=big), 0.9),
        Match(_rec("b2", tmp_path, body=big), 0.9),
    ]
    res = build_injection(
        m, policy=InjectionPolicy(max_skills=2, total_char_budget=6000)
    )
    # second body would blow the budget, so only the first is kept
    assert len(res.injected_keys) == 1


def test_pure_semantic_without_keywords_filtered(tmp_path):
    # hybrid=0.35, cosine=0.35 → zero lexical boost, cosine < 0.50 → gate rejects
    m = [Match(_rec("epsilon", tmp_path), 0.35, cosine=0.35)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "none"
    assert res.text is None


def test_strong_cosine_alone_passes_gate(tmp_path):
    # hybrid=0.55, cosine=0.55 → no lexical boost, but cosine >= 0.50 → passes
    m = [Match(_rec("zeta", tmp_path), 0.55, cosine=0.55)]
    res = build_injection(m, policy=InjectionPolicy())
    assert res.mode == "body"
    assert "BODY-CONTENT" in res.text
