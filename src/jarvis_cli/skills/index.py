"""Build, persist, and load the skill embedding index.

The index is a sidecar of the catalog: `catalog.json` (records + model name +
per-skill content hashes) and `vectors.npy` (row-aligned embeddings). It lives
under `~/.jarvis-cli/skills/` and is rebuilt only when the on-disk skills change
(by `content_hash`) or the model changes — so the launchd daemon pays the embed
cost once, not per prompt.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from .catalog import SkillRecord
from .embedder import Embedder

_CATALOG_NAME = "catalog.json"
_VECTORS_NAME = "vectors.npy"


@dataclass
class SkillIndex:
    model_name: str
    records: list[SkillRecord]
    vectors: np.ndarray  # shape (len(records), dim), L2-normalized

    def signature(self) -> dict[str, str]:
        """key -> content_hash, for cheap staleness checks against a fresh scan."""
        return {r.key(): r.content_hash for r in self.records}


def build_index(records: list[SkillRecord], embedder: Embedder) -> SkillIndex:
    texts = [r.text_for_embedding() for r in records]
    vectors = embedder.embed(texts) if texts else np.empty((0, 0), dtype=np.float32)
    logger.info("skills: built index for {} skills", len(records))
    return SkillIndex(model_name=embedder.model_name, records=records, vectors=vectors)


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


def save_index(index: SkillIndex, index_dir: Path) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": index.model_name,
        "records": [_record_to_dict(r) for r in index.records],
    }
    (index_dir / _CATALOG_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(index_dir / _VECTORS_NAME, index.vectors)


def load_index(index_dir: Path) -> SkillIndex | None:
    index_dir = Path(index_dir)
    cat = index_dir / _CATALOG_NAME
    vec = index_dir / _VECTORS_NAME
    if not cat.exists() or not vec.exists():
        return None
    try:
        payload = json.loads(cat.read_text(encoding="utf-8"))
        vectors = np.load(vec)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("skills: failed to load index ({}); will rebuild", exc)
        return None
    records = [_record_from_dict(d) for d in payload.get("records", [])]
    if len(records) != vectors.shape[0]:
        logger.warning("skills: index row mismatch; will rebuild")
        return None
    return SkillIndex(
        model_name=payload.get("model_name", ""), records=records, vectors=vectors
    )


def is_stale(index: SkillIndex, fresh: list[SkillRecord], model_name: str) -> bool:
    """True if the index no longer matches the on-disk skills or the model."""
    if index.model_name != model_name:
        return True
    return index.signature() != {r.key(): r.content_hash for r in fresh}


def ensure_index(
    index_dir: Path, embedder: Embedder, fresh: list[SkillRecord]
) -> SkillIndex:
    """Load a current index from disk, or (re)build and persist it when the
    on-disk skills or the model changed."""
    existing = load_index(index_dir)
    if existing is not None and not is_stale(existing, fresh, embedder.model_name):
        logger.debug("skills: index up to date ({} skills)", len(existing.records))
        return existing
    rebuilt = build_index(fresh, embedder)
    save_index(rebuilt, index_dir)
    return rebuilt
