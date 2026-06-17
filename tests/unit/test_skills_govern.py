import json
from pathlib import Path

from jarvis_cli.skills.catalog import scan_skills
from jarvis_cli.skills.govern import (
    GovernPaths,
    apply_governance,
    build_plan,
    restore_governance,
)


def _skill(path: Path, name: str, desc: str = "d") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody", encoding="utf-8")


def _fake_home(tmp_path: Path) -> Path:
    # standalone skills
    _skill(tmp_path / ".claude" / "skills" / "git-commit" / "SKILL.md", "git-commit")
    _skill(tmp_path / ".claude" / "skills" / "continue" / "SKILL.md", "continue")
    # a skill-providing plugin with one agent
    root = tmp_path / ".claude" / "plugins" / "cache" / "mkt" / "superpowers" / "5.0.0"
    _skill(root / "skills" / "brainstorming" / "SKILL.md", "brainstorming")
    (root / "agents").mkdir(parents=True)
    (root / "agents" / "code-reviewer.md").write_text(
        "---\nname: code-reviewer\n---\nreview", encoding="utf-8"
    )
    # a non-skill plugin (no SKILL.md) — must be left enabled
    (tmp_path / ".claude" / "plugins" / "cache" / "mkt2" / "swift-lsp" / "1.0.0").mkdir(
        parents=True
    )
    # settings.json with all plugins enabled
    settings = tmp_path / ".claude" / "settings.json"
    settings.write_text(json.dumps({
        "enabledPlugins": {
            "superpowers@mkt": True,
            "swift-lsp@mkt2": True,
        }
    }), encoding="utf-8")
    return tmp_path


def _paths(home: Path) -> GovernPaths:
    return GovernPaths.default(home=home)


def test_build_plan_targets_standalone_and_skill_plugins(tmp_path):
    home = _fake_home(tmp_path)
    records = scan_skills(home=home)
    enabled = json.loads((home / ".claude" / "settings.json").read_text())["enabledPlugins"]
    plan = build_plan(records, enabled, cc_agents=home / ".claude" / "agents")
    assert set(plan.standalone) == {"git-commit", "continue"}
    assert plan.plugins == ["superpowers@mkt"]  # swift-lsp has no skills, excluded
    assert [Path(d).name for _, d in plan.agents] == ["code-reviewer.md"]


def test_keep_excludes_from_plan(tmp_path):
    home = _fake_home(tmp_path)
    records = scan_skills(home=home)
    enabled = json.loads((home / ".claude" / "settings.json").read_text())["enabledPlugins"]
    plan = build_plan(records, enabled, keep={"git-commit", "superpowers"},
                      cc_agents=home / ".claude" / "agents")
    assert plan.standalone == ["continue"]
    assert plan.plugins == []  # superpowers kept
    assert plan.agents == []   # kept plugin's agent not re-homed


def test_apply_then_restore_roundtrip(tmp_path):
    home = _fake_home(tmp_path)
    paths = _paths(home)
    records = scan_skills(home=home)
    enabled = json.loads(paths.cc_settings.read_text())["enabledPlugins"]
    plan = build_plan(records, enabled, cc_agents=paths.cc_agents)

    apply_governance(plan, paths)
    # skillOverrides written
    local = json.loads(paths.cc_settings_local.read_text())
    assert local["skillOverrides"]["git-commit"] == "user-invocable-only"
    # plugin disabled, non-skill plugin untouched
    settings = json.loads(paths.cc_settings.read_text())
    assert settings["enabledPlugins"]["superpowers@mkt"] is False
    assert settings["enabledPlugins"]["swift-lsp@mkt2"] is True
    # agent re-homed
    assert (paths.cc_agents / "code-reviewer.md").exists()
    assert paths.manifest.exists()

    res = restore_governance(paths)
    assert res["restored"] is True
    settings = json.loads(paths.cc_settings.read_text())
    assert settings["enabledPlugins"]["superpowers@mkt"] is True  # re-enabled
    local = json.loads(paths.cc_settings_local.read_text())
    assert "skillOverrides" not in local  # all ours removed
    assert not (paths.cc_agents / "code-reviewer.md").exists()  # removed
    assert not paths.manifest.exists()


def test_apply_preserves_other_settings_and_user_overrides(tmp_path):
    home = _fake_home(tmp_path)
    paths = _paths(home)
    # user has their own setting + their own skillOverride
    settings = json.loads(paths.cc_settings.read_text())
    settings["theme"] = "dark"
    paths.cc_settings.write_text(json.dumps(settings))
    paths.cc_settings_local.write_text(json.dumps({"skillOverrides": {"mine": "off"}}))

    records = scan_skills(home=home)
    enabled = json.loads(paths.cc_settings.read_text())["enabledPlugins"]
    plan = build_plan(records, enabled, cc_agents=paths.cc_agents)
    apply_governance(plan, paths)
    restore_governance(paths)

    # user's settings survive the round-trip
    assert json.loads(paths.cc_settings.read_text())["theme"] == "dark"
    assert json.loads(paths.cc_settings_local.read_text())["skillOverrides"] == {"mine": "off"}


def test_restore_without_manifest_is_noop(tmp_path):
    res = restore_governance(_paths(_fake_home(tmp_path)))
    assert res["restored"] is False


def test_apply_does_not_overwrite_existing_agent(tmp_path):
    home = _fake_home(tmp_path)
    paths = _paths(home)
    paths.cc_agents.mkdir(parents=True)
    (paths.cc_agents / "code-reviewer.md").write_text("USER VERSION", encoding="utf-8")
    records = scan_skills(home=home)
    enabled = json.loads(paths.cc_settings.read_text())["enabledPlugins"]
    plan = build_plan(records, enabled, cc_agents=paths.cc_agents)
    apply_governance(plan, paths)
    # user's agent preserved; not in rehomed-now list (so restore won't delete it)
    assert (paths.cc_agents / "code-reviewer.md").read_text() == "USER VERSION"
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["agents_rehomed_now"] == []
