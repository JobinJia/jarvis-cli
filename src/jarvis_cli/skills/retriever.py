"""Skill retriever — re-exports the generic retriever under domain names.

All retrieval logic lives in ``retrieval.retriever``; this module keeps
existing import paths (``from jarvis_cli.skills.retriever import ...``)
working.
"""
from ..retrieval.retriever import Match, Retriever as SkillRetriever

__all__ = ["Match", "SkillRetriever"]
