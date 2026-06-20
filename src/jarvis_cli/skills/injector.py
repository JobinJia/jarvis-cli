"""Turn retrieval results into the text injected as `additionalContext`.

Gated by signal structure, not just a score threshold:
  * A match must have lexical signal (keyword/name overlap with the query)
    OR very strong cosine (>= 0.50) to pass the gate at all — pure semantic
    near-misses are filtered out regardless of score.
  * Among gated matches: strong -> inject the full skill body; medium ->
    inject a one-line menu; weak -> nothing.

Bodies already injected earlier in the session are skipped: CC keeps an
injected message for the rest of the session, so re-injecting is wasted
context. Callers pass the running `already_injected` set and merge the result's
`injected_keys` back into it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .loader import load_body
from .retriever import Match

_COSINE_SOLO_FLOOR = 0.50


def _has_lexical_signal(m: Match) -> bool:
    """True when the keyword / name boost contributed to the hybrid score."""
    return m.score - m.cosine > 1e-4


@dataclass
class InjectionPolicy:
    high_threshold: float = 0.42
    med_threshold: float = 0.28
    max_skills: int = 2  # max bodies injected in one turn
    max_body_chars: int = 6000  # per-skill body cap (~first chunk)
    total_char_budget: int = 9000  # across all bodies this turn


@dataclass
class InjectionResult:
    text: str | None  # additionalContext, or None to inject nothing
    mode: str  # "body" | "menu" | "none"
    injected_keys: list[str] = field(default_factory=list)


_BODY_PREAMBLE = (
    "Relevant skill(s) auto-loaded for this request. Treat each as standing "
    "instructions and follow it if it applies (you do not need the Skill "
    "tool):\n\n"
)
_MENU_PREAMBLE = (
    "Possibly relevant skills — load one with its slash command or the Skill "
    "tool if it fits:\n"
)


def _oneline(text: str, limit: int = 120) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def build_injection(
    matches: list[Match],
    *,
    policy: InjectionPolicy | None = None,
    already_injected: set[str] | None = None,
) -> InjectionResult:
    policy = policy or InjectionPolicy()
    already = already_injected or set()

    gated = [
        m for m in matches
        if m.score >= policy.med_threshold
        and (_has_lexical_signal(m) or m.cosine >= _COSINE_SOLO_FLOOR)
    ]
    if not gated:
        return InjectionResult(text=None, mode="none")

    strong = [m for m in gated if m.score >= policy.high_threshold]
    strong_fresh = [m for m in strong if m.record.key() not in already]

    if strong_fresh:
        chunks: list[str] = []
        keys: list[str] = []
        used = 0
        for m in strong_fresh[: policy.max_skills]:
            body = load_body(m.record, max_chars=policy.max_body_chars)
            if body is None:
                continue
            if used + len(body) > policy.total_char_budget and chunks:
                break
            chunks.append(body)
            keys.append(m.record.key())
            used += len(body)
        if chunks:
            return InjectionResult(
                text=_BODY_PREAMBLE + "\n\n---\n\n".join(chunks),
                mode="body",
                injected_keys=keys,
            )

    # Strong matches all already in context this session -> nothing to add.
    if strong and not strong_fresh:
        return InjectionResult(text=None, mode="none")

    # Only medium-confidence matches: offer a cheap menu instead of a body.
    lines = [
        f"- `{m.record.name}` — {_oneline(m.record.description)}"
        for m in gated[: max(policy.max_skills + 1, 3)]
    ]
    return InjectionResult(text=_MENU_PREAMBLE + "\n".join(lines), mode="menu")
