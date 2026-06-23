"""Skill retriever — re-exports the generic retriever under domain names.

All retrieval logic lives in ``retrieval.retriever``; this module keeps
existing import paths (``from jarvis_cli.skills.retriever import ...``)
working.
"""
from ..retrieval.retriever import (
    Match,
    Retriever as SkillRetriever,
    gate_matches,
    has_lexical_signal,
)

__all__ = ["Match", "SkillRetriever", "gate_matches", "has_lexical_signal"]
