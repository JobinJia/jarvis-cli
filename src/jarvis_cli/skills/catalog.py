"""Discover installed skills across Claude Code and Codex, no heavy deps.

A skill is any `SKILL.md` with YAML frontmatter carrying at least a `name`.
We scan every place CC and Codex load skills from, plus plugin caches (whose
skills can't be hidden per-skill via `skillOverrides`, so the hook must be able
to surface them even when the plugin is disabled — see the project notes).

The scan deliberately reads files that may be hidden from the startup prompt or
belong to a disabled plugin: injection reads the body directly, so a skill's
reachability here is independent of whether CC/Codex would list it.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from ..retrieval.text import deslug, lexical_tokens

# Skill source roots (relative to home, `~` expanded at scan time). We walk
# each tree and keep any `SKILL.md` whose grandparent dir is `skills` — i.e.
# `…/skills/<name>/SKILL.md`. That single rule covers every layout seen in the
# wild: standalone `~/.claude/skills/<name>/`, plugin `…/<ver>/skills/<name>/`,
# and the `…/<ver>/.claude/skills/<name>/` variant some plugins (ui-ux-pro-max)
# use — without matching bundled example/SKILL.md files nested elsewhere.
_USER_ROOTS: tuple[tuple[str, str], ...] = (
    ("claude", ".claude/skills"),
    ("claude", ".claude/plugins/cache"),
    ("codex", ".codex/skills"),
)


def _is_skill_md(path: Path) -> bool:
    return path.name == "SKILL.md" and path.parent.parent.name == "skills"


@dataclass
class SkillRecord:
    """One discovered skill. `text_for_embedding` is what the index vectorizes;
    `body_path` is what the loader reads to inject."""

    name: str
    description: str
    source_tool: str  # "claude" | "codex"
    path: str  # absolute path to SKILL.md
    when_to_use: str = ""
    keywords: list[str] = field(default_factory=list)
    plugin: str = ""  # non-empty when the skill comes from a plugin cache
    body_chars: int = 0
    content_hash: str = ""  # frontmatter signature; cheap change detection

    def text_for_embedding(self) -> str:
        # De-slugify the name ("deploy-to-vercel" -> "deploy to vercel",
        # "ckm:ui-styling" -> "ui styling") so the model sees real words, and
        # repeat it: the name is the single most discriminative signal but is
        # short, so weighting it counteracts long generic descriptions that
        # otherwise dominate the vector.
        words = deslug(self.name)
        parts = [words, words, self.description, self.when_to_use]
        if self.keywords:
            parts.append(" ".join(self.keywords))
        return "\n".join(p for p in parts if p).strip()

    def key(self) -> str:
        """Stable identity for dedup across roots (a symlinked skill can be
        reached by two globs). Plugin-qualified so two plugins can ship a
        skill of the same name without colliding."""
        return f"{self.plugin}/{self.name}" if self.plugin else self.name


def _split_frontmatter(text: str) -> tuple[dict, int]:
    """Return (frontmatter_dict, body_char_count). Tolerates a missing or
    malformed block by returning ({}, len(text))."""
    if not text.startswith("---"):
        return {}, len(text)
    end = text.find("\n---", 3)
    if end == -1:
        return {}, len(text)
    raw = text[3:end]
    body = text[end + 4 :]
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}, len(body)
    if not isinstance(data, dict):
        return {}, len(body)
    return data, len(body)


def _coerce_keywords(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [k.strip() for k in value.replace(",", " ").split() if k.strip()]
    return []


def _plugin_name(path: Path) -> str:
    """Derive the plugin slug from a plugins/cache path, else ''. Layout is
    `.../plugins/cache/<marketplace>/<plugin>/<version>/skills/<name>/SKILL.md`."""
    parts = path.parts
    if "cache" in parts:
        i = parts.index("cache")
        # marketplace at i+1, plugin at i+2
        if i + 2 < len(parts):
            return parts[i + 2]
    return ""


def parse_skill_file(path: Path, source_tool: str) -> SkillRecord | None:
    """Parse one SKILL.md into a record, or None if it has no usable name."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("skills: unreadable {} ({})", path, exc)
        return None
    fm, body_chars = _split_frontmatter(text)
    name = str(fm.get("name") or path.parent.name).strip()
    if not name:
        return None
    description = str(fm.get("description") or "").strip()
    # `when_to_use` (snake) is the Agent Skills field; tolerate the kebab spelling.
    when = str(fm.get("when_to_use") or fm.get("when-to-use") or "").strip()
    keywords = _coerce_keywords(fm.get("keywords"))
    sig = f"{name}\0{description}\0{when}\0{','.join(keywords)}"
    return SkillRecord(
        name=name,
        description=description,
        source_tool=source_tool,
        path=str(path),
        when_to_use=when,
        keywords=keywords,
        plugin=_plugin_name(path),
        body_chars=body_chars,
        content_hash=hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16],
    )


def scan_skills(
    *,
    home: Path | None = None,
    extra_roots: list[tuple[str, Path]] | None = None,
) -> list[SkillRecord]:
    """Discover all skills. `extra_roots` is a list of (source_tool, dir) for
    per-project skill dirs (e.g. a repo's `.codex/skills`). Deduplicates by
    `SkillRecord.key()`, keeping the first hit (user roots win over plugins)."""
    home = home or Path.home()
    seen: dict[str, SkillRecord] = {}

    def _consider(rec: SkillRecord | None) -> None:
        if rec is None:
            return
        seen.setdefault(rec.key(), rec)

    visited: set[str] = set()

    def _scan_root(source_tool: str, root: Path) -> None:
        if not root.is_dir():
            return
        # os.walk with followlinks=True so symlinked skill dirs (common in
        # ~/.claude/skills, e.g. a skill linked from another repo) are scanned —
        # plain rglob skips symlinked directories. `visited` (by realpath)
        # guards against symlink cycles.
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            real = os.path.realpath(dirpath)
            if real in visited:
                dirnames[:] = []
                continue
            visited.add(real)
            if "SKILL.md" in filenames:
                p = Path(dirpath) / "SKILL.md"
                if _is_skill_md(p):
                    _consider(parse_skill_file(p.resolve(), source_tool))

    for source_tool, rel in _USER_ROOTS:
        _scan_root(source_tool, home / rel)

    for source_tool, root in extra_roots or []:
        _scan_root(source_tool, Path(root))

    records = list(seen.values())
    logger.debug("skills: scanned {} unique skills", len(records))
    return records
