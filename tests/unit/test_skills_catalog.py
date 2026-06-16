import os
from pathlib import Path

from jarvis_cli.skills.catalog import (
    deslug,
    lexical_tokens,
    parse_skill_file,
    scan_skills,
)


def test_deslug_splits_separators_and_camel():
    assert deslug("deploy-to-vercel") == "deploy to vercel"
    assert deslug("ckm:ui-styling") == "ckm ui styling"
    assert deslug("writeTests") == "write Tests"


def test_lexical_tokens_ascii_and_cjk():
    toks = lexical_tokens("deploy a Vercel 部署上线")
    assert "deploy" in toks
    assert "vercel" in toks  # lowercased
    assert "部署上线" in toks  # CJK run
    assert "a" not in toks  # single ASCII char dropped (min len 2)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


_FM = """---
name: demo-skill
description: A demo skill for testing
when_to_use: when you are testing
keywords: [alpha, beta]
---

# Body
Do the thing.
"""


def test_parse_frontmatter_fields(tmp_path):
    p = _write(tmp_path / "skills" / "demo-skill" / "SKILL.md", _FM)
    rec = parse_skill_file(p, "claude")
    assert rec is not None
    assert rec.name == "demo-skill"
    assert rec.description == "A demo skill for testing"
    assert rec.when_to_use == "when you are testing"
    assert rec.keywords == ["alpha", "beta"]
    assert rec.body_chars > 0
    # name is de-slugged into words for embedding ("demo-skill" -> "demo skill")
    assert "demo skill" in rec.text_for_embedding()
    assert "alpha" in rec.text_for_embedding()


def test_keywords_accepts_comma_string(tmp_path):
    fm = "---\nname: k\ndescription: d\nkeywords: one, two three\n---\nbody\n"
    p = _write(tmp_path / "skills" / "k" / "SKILL.md", fm)
    rec = parse_skill_file(p, "claude")
    assert rec is not None
    assert rec.keywords == ["one", "two", "three"]


def test_missing_name_falls_back_to_dir(tmp_path):
    p = _write(tmp_path / "skills" / "named-by-dir" / "SKILL.md", "---\ndescription: d\n---\nx")
    rec = parse_skill_file(p, "claude")
    assert rec is not None
    assert rec.name == "named-by-dir"


def test_malformed_frontmatter_is_tolerated(tmp_path):
    p = _write(tmp_path / "skills" / "broken" / "SKILL.md", "no frontmatter here")
    rec = parse_skill_file(p, "claude")
    assert rec is not None
    assert rec.name == "broken"
    assert rec.description == ""


def test_scan_finds_standalone_plugin_and_nested_claude_layouts(tmp_path):
    # standalone
    _write(tmp_path / ".claude" / "skills" / "solo" / "SKILL.md",
           "---\nname: solo\ndescription: standalone\n---\nb")
    # plugin: .../<ver>/skills/<name>/
    _write(
        tmp_path / ".claude" / "plugins" / "cache" / "mkt" / "plug" / "1.0.0"
        / "skills" / "plugskill" / "SKILL.md",
        "---\nname: plugskill\ndescription: from plugin\n---\nb",
    )
    # plugin: .../<ver>/.claude/skills/<name>/  (ui-ux-pro-max style)
    _write(
        tmp_path / ".claude" / "plugins" / "cache" / "mkt2" / "plug2" / "2.5.0"
        / ".claude" / "skills" / "nested" / "SKILL.md",
        "---\nname: nested\ndescription: nested layout\n---\nb",
    )
    # codex
    _write(tmp_path / ".codex" / "skills" / "cx" / "SKILL.md",
           "---\nname: cx\ndescription: codex skill\n---\nb")
    # a non-skill SKILL.md nested elsewhere must be ignored
    _write(tmp_path / ".claude" / "skills" / "solo" / "examples" / "ex" / "SKILL.md",
           "---\nname: should-ignore\ndescription: x\n---\nb")

    recs = {r.name: r for r in scan_skills(home=tmp_path)}
    assert set(recs) == {"solo", "plugskill", "nested", "cx"}
    assert recs["plugskill"].plugin == "plug"
    assert recs["nested"].plugin == "plug2"
    assert recs["cx"].source_tool == "codex"
    assert recs["solo"].plugin == ""


def test_scan_follows_symlinked_skill_dirs(tmp_path):
    # A skill whose dir under ~/.claude/skills is a symlink to a real dir
    # elsewhere (common: linking a skill from another repo). rglob skips these;
    # os.walk(followlinks=True) must pick it up.
    real = tmp_path / "elsewhere" / "linked-skill"
    real.mkdir(parents=True)
    (real / "SKILL.md").write_text(
        "---\nname: linked-skill\ndescription: via symlink\n---\nbody", encoding="utf-8"
    )
    skills_dir = tmp_path / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    os.symlink(real, skills_dir / "linked-skill")

    recs = {r.name for r in scan_skills(home=tmp_path)}
    assert "linked-skill" in recs


def test_scan_dedups_by_key(tmp_path):
    # same skill reachable twice (simulate symlink by two identical roots is
    # hard; instead assert two different-named skills don't collide and a
    # repeated name under different plugins stays distinct)
    _write(
        tmp_path / ".claude" / "plugins" / "cache" / "m" / "p1" / "1" / "skills"
        / "dup" / "SKILL.md", "---\nname: dup\ndescription: one\n---\nb")
    _write(
        tmp_path / ".claude" / "plugins" / "cache" / "m" / "p2" / "1" / "skills"
        / "dup" / "SKILL.md", "---\nname: dup\ndescription: two\n---\nb")
    recs = scan_skills(home=tmp_path)
    keys = sorted(r.key() for r in recs)
    assert keys == ["p1/dup", "p2/dup"]
