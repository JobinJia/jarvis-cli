"""Skill-specific embedding index: thin layer over the generic retrieval index.

Provides ``SkillRecord`` serializers and convenience wrappers that bind them
so callers (SkillService, CLI) keep the same call signatures as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..retrieval.embedder import Embedder
from ..retrieval.index import Index as _Index
from ..retrieval.index import (
    build_index,
    ensure_index as _ensure_index,
    is_stale,
    load_index as _load_index,
    save_index as _save_index,
)
from .catalog import SkillRecord


@dataclass
class SkillIndex:
    model_name: str
    records: list[SkillRecord]
    vectors: np.ndarray

    def signature(self) -> dict[str, str]:
        return {r.key(): r.content_hash for r in self.records}


def _record_to_dict(r: SkillRecord) -> dict:
    return {
        "name": r.name,
        "description": r.description,
        "source_tool": r.source_tool,
        "path": r.path,
        "when_to_use": r.when_to_use,
        "keywords": r.keywords,
        "plugin": r.plugin,
        "body_chars": r.body_chars,
        "content_hash": r.content_hash,
    }


def _record_from_dict(d: dict) -> SkillRecord:
    return SkillRecord(
        name=d["name"],
        description=d.get("description", ""),
        source_tool=d.get("source_tool", "claude"),
        path=d["path"],
        when_to_use=d.get("when_to_use", ""),
        keywords=list(d.get("keywords", [])),
        plugin=d.get("plugin", ""),
        body_chars=int(d.get("body_chars", 0)),
        content_hash=d.get("content_hash", ""),
    )


def _to_skill_index(idx: _Index) -> SkillIndex:
    return SkillIndex(
        model_name=idx.model_name, records=idx.records, vectors=idx.vectors,
    )


def save_index(index: SkillIndex, index_dir: Path) -> None:
    _save_index(
        _Index(index.model_name, index.records, index.vectors),
        index_dir,
        _record_to_dict,
    )


def load_index(index_dir: Path) -> SkillIndex | None:
    idx = _load_index(index_dir, _record_from_dict)
    return _to_skill_index(idx) if idx is not None else None


def ensure_index(
    index_dir: Path, embedder: Embedder, fresh: list[SkillRecord]
) -> SkillIndex:
    idx = _ensure_index(
        index_dir, embedder, fresh, _record_to_dict, _record_from_dict,
    )
    return _to_skill_index(idx)
