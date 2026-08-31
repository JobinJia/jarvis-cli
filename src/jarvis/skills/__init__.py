"""RAG-over-skills: bound CC/Codex startup context while keeping every
installed skill reachable.

Skill descriptions cost startup context whether or not a skill is ever used.
This subpackage moves the long tail out of the startup prompt and surfaces the
right skill per-turn instead: a `UserPromptSubmit` hook embeds the user's
prompt, retrieves the closest skills from a local index, and injects the
matching skill body as `additionalContext`. The embedding model lives in the
resident daemon so the hook stays sub-50ms.

Everything that touches the embedding model (`index`, `retriever`) imports
`fastembed` lazily so the base install (no `skills` extra) keeps working —
`catalog` and `loader` have no heavy deps and are always importable.
"""
from __future__ import annotations
