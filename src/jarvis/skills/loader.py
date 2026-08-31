"""Read a skill's body for injection.

We inject the rendered body as `additionalContext` rather than invoking the
Skill tool, so this works for skills hidden from the startup list or living in a
disabled plugin. Bundled-resource placeholders (`${CLAUDE_SKILL_DIR}`) are not
resolved by CC in injected context, so we prepend the skill directory and tell
the model that path resolves the placeholder.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from .catalog import SkillRecord


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    if end == -1:
        return text
    return text[end + 4 :].lstrip("\n")


def load_body(record: SkillRecord, *, max_chars: int = 0) -> str | None:
    """Return the skill body (frontmatter stripped) with a small header that
    names the skill and its directory. None if the file vanished. `max_chars`
    > 0 truncates the body (CC keeps the first ~5k tokens of an invoked skill;
    we cap injected bodies similarly to stay bounded)."""
    path = Path(record.path)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("skills: body unreadable {} ({})", path, exc)
        return None
    body = _strip_frontmatter(raw).strip()
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n\n…(skill body truncated)"
    skill_dir = path.parent
    header = (
        f"### Skill: {record.name}\n"
        f"(auto-loaded by relevance; skill dir = {skill_dir}, "
        f"which is what ${{CLAUDE_SKILL_DIR}} / bundled paths refer to)\n"
    )
    return header + "\n" + body
