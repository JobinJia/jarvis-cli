from pathlib import Path

from jarvis_cli.skills.catalog import SkillRecord
from jarvis_cli.skills.loader import load_body


def _rec(tmp: Path, text: str) -> SkillRecord:
    p = tmp / "skills" / "demo" / "SKILL.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return SkillRecord(name="demo", description="d", source_tool="claude", path=str(p))


def test_load_body_strips_frontmatter_and_adds_header(tmp_path):
    rec = _rec(tmp_path, "---\nname: demo\ndescription: d\n---\n\nReal body here.")
    body = load_body(rec)
    assert body is not None
    assert "### Skill: demo" in body
    assert "Real body here." in body
    assert "description: d" not in body  # frontmatter removed


def test_load_body_truncates(tmp_path):
    rec = _rec(tmp_path, "---\nname: demo\n---\n" + "A" * 1000)
    body = load_body(rec, max_chars=100)
    assert body is not None
    assert "truncated" in body


def test_load_body_missing_file_returns_none(tmp_path):
    rec = SkillRecord(name="x", description="d", source_tool="claude",
                      path=str(tmp_path / "nope" / "SKILL.md"))
    assert load_body(rec) is None
