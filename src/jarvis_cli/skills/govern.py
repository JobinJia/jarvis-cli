"""Codify the skill-hiding policy as a repeatable, reversible operation.

Retrieval already covers plugin skills (the catalog scans plugin caches
regardless of whether a plugin is enabled). What this module manages is the
*lifecycle* side that otherwise has to be hand-edited across config files:

  * standalone skills (`~/.claude/skills/`) -> `skillOverrides` in
    settings.local.json, so their description leaves the model's startup
    context while `/name` still works.
  * skill-providing plugins -> disabled in `enabledPlugins` (the only way to
    drop a plugin skill's description; `skillOverrides` does not touch plugin
    skills), with their agents re-homed to `~/.claude/agents/` so a disabled
    plugin's non-skill pieces (e.g. superpowers' code-reviewer) stay reachable.

`apply` records exactly what it changed to a manifest; `restore` reverses it
from that manifest. Non-skill plugins (no SKILL.md) are left untouched —
disabling them buys no context and would break their features.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .catalog import SkillRecord, scan_skills

MANIFEST_VERSION = 1
# settings.local.json visibility state: hidden from the model, still /-invocable.
DEFAULT_MODE = "user-invocable-only"


@dataclass
class GovernPaths:
    """All filesystem locations the governor reads/writes. Injectable so tests
    run fully against a tmp home."""

    home: Path
    cc_settings: Path
    cc_settings_local: Path
    cc_agents: Path
    manifest: Path

    @classmethod
    def default(cls, home: Path | None = None) -> GovernPaths:
        home = home or Path.home()
        return cls(
            home=home,
            cc_settings=home / ".claude" / "settings.json",
            cc_settings_local=home / ".claude" / "settings.local.json",
            cc_agents=home / ".claude" / "agents",
            manifest=home / ".jarvis-cli" / "skills" / "govern-manifest.json",
        )


@dataclass
class GovernPlan:
    mode: str
    standalone: list[str] = field(default_factory=list)  # skill names to hide
    plugins: list[str] = field(default_factory=list)  # enabledPlugins keys
    agents: list[tuple[str, str]] = field(default_factory=list)  # (src, dst)
    codex_skills: list[str] = field(default_factory=list)  # reported, see below

    def is_empty(self) -> bool:
        return not (self.standalone or self.plugins or self.agents)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("govern: unreadable {} ({})", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _plugin_root(record: SkillRecord) -> Path | None:
    """`…/plugins/cache/<mkt>/<plugin>/<ver>/…/SKILL.md` -> the `<ver>` dir."""
    parts = Path(record.path).parts
    if "cache" not in parts:
        return None
    i = parts.index("cache")
    if i + 4 > len(parts):
        return None
    return Path(*parts[: i + 4])


def build_plan(
    records: list[SkillRecord],
    enabled_plugins: dict[str, bool],
    *,
    keep: set[str] | None = None,
    mode: str = DEFAULT_MODE,
    cc_agents: Path | None = None,
) -> GovernPlan:
    """Compute the governance actions. `keep` is a hot-set of skill names and/or
    plugin slugs to leave fully visible."""
    keep = keep or set()
    plan = GovernPlan(mode=mode)

    # Standalone Claude skills (no plugin) -> hide unless kept.
    plan.standalone = sorted(
        r.name
        for r in records
        if r.source_tool == "claude" and not r.plugin and r.name not in keep
    )

    # Plugin slugs that ship at least one skill -> disable unless kept.
    skill_plugin_slugs = {
        r.plugin for r in records if r.plugin and r.plugin not in keep
    }
    # Map slug -> the "<plugin>@<marketplace>" key used in enabledPlugins.
    for key in enabled_plugins:
        slug = key.split("@", 1)[0]
        if slug in skill_plugin_slugs:
            plan.plugins.append(key)
    plan.plugins.sort()

    # Re-home each disabled plugin's agents so they survive the disable.
    seen_roots: set[Path] = set()
    for r in records:
        if not r.plugin or r.plugin in keep:
            continue
        root = _plugin_root(r)
        if root is None or root in seen_roots:
            continue
        seen_roots.add(root)
        agents_dir = root / "agents"
        if not agents_dir.is_dir():
            continue
        for agent in sorted(agents_dir.glob("*.md")):
            dst = (cc_agents or Path.home() / ".claude" / "agents") / agent.name
            plan.agents.append((str(agent), str(dst)))

    # Codex skills are scanned the same way; report them (none installed yet).
    plan.codex_skills = sorted(r.name for r in records if r.source_tool == "codex")
    return plan


def apply_governance(plan: GovernPlan, paths: GovernPaths) -> dict:
    """Apply the plan and write the manifest. Idempotent: re-running with the
    same plan is a no-op beyond rewriting the manifest."""
    # 1. Hide standalone skills via skillOverrides.
    local = _read_json(paths.cc_settings_local)
    overrides = local.setdefault("skillOverrides", {})
    for name in plan.standalone:
        overrides[name] = plan.mode
    _write_json(paths.cc_settings_local, local)

    # 2. Disable skill-providing plugins.
    settings = _read_json(paths.cc_settings)
    ep = settings.setdefault("enabledPlugins", {})
    for key in plan.plugins:
        ep[key] = False
    _write_json(paths.cc_settings, settings)

    # 3. Re-home agents (never overwrite a user's existing agent).
    rehomed: list[str] = []
    paths.cc_agents.mkdir(parents=True, exist_ok=True)
    for src, dst in plan.agents:
        dpath = Path(dst)
        if dpath.exists():
            logger.debug("govern: agent already present, skipping {}", dpath)
            continue
        shutil.copy2(src, dpath)
        rehomed.append(dst)

    manifest = {
        "version": MANIFEST_VERSION,
        "mode": plan.mode,
        "skillOverrides": plan.standalone,
        "plugins": plan.plugins,
        "agents": [dst for _, dst in plan.agents],
        "agents_rehomed_now": rehomed,
    }
    _write_json(paths.manifest, manifest)
    return manifest


def restore_governance(paths: GovernPaths) -> dict:
    """Reverse a prior `apply` from the manifest: remove the skillOverrides we
    set, re-enable the plugins we disabled, delete the agents we re-homed. A
    skill-providing plugin's ungoverned state is enabled, so restore re-enables
    all managed plugins."""
    manifest = _read_json(paths.manifest)
    if not manifest:
        return {"restored": False, "reason": "no manifest"}

    # 1. Drop our skillOverrides (only those still set to our mode).
    local = _read_json(paths.cc_settings_local)
    overrides = local.get("skillOverrides", {})
    mode = manifest.get("mode", DEFAULT_MODE)
    removed_overrides = []
    for name in manifest.get("skillOverrides", []):
        if overrides.get(name) == mode:
            del overrides[name]
            removed_overrides.append(name)
    if not overrides:
        local.pop("skillOverrides", None)
    _write_json(paths.cc_settings_local, local)

    # 2. Re-enable the plugins we disabled.
    settings = _read_json(paths.cc_settings)
    ep = settings.get("enabledPlugins", {})
    reenabled = []
    for key in manifest.get("plugins", []):
        if key in ep:
            ep[key] = True
            reenabled.append(key)
    _write_json(paths.cc_settings, settings)

    # 3. Remove only the agents we re-homed (this run's copies).
    removed_agents = []
    for dst in manifest.get("agents_rehomed_now", []):
        p = Path(dst)
        if p.exists():
            p.unlink()
            removed_agents.append(dst)

    paths.manifest.unlink(missing_ok=True)
    return {
        "restored": True,
        "skillOverrides_removed": removed_overrides,
        "plugins_reenabled": reenabled,
        "agents_removed": removed_agents,
    }


def discover(home: Path | None = None) -> list[SkillRecord]:
    return scan_skills(home=home)
