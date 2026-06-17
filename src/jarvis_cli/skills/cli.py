"""`jarvis-cli skills <status|index|query>` — operate/debug the skill index.

`status` is dependency-light (catalog scan only); `index` and `query` load the
embedding model, so they require the `skills` extra and print a clear hint if
it's missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DEFAULT_CONFIG_PATH, load_config
from .catalog import scan_skills


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("skills", help="manage the RAG-over-skills index")
    ssub = p.add_subparsers(dest="skills_cmd", required=True)
    ssub.add_parser("status", help="list discovered skills (no model load)")
    ssub.add_parser("index", help="(re)build the embedding index")
    dl = ssub.add_parser(
        "download", help="pre-fetch the embedding model (resumable; slow nets)"
    )
    dl.add_argument(
        "--attempts", type=int, default=40,
        help="max resume attempts before giving up (default 40)",
    )
    q = ssub.add_parser("query", help="show what a prompt would retrieve")
    q.add_argument("text", nargs="+", help="the prompt text to test")
    p.set_defaults(func=cmd_skills)


def _hf_repo_for(model_name: str) -> str | None:
    """Resolve the HF repo fastembed downloads `model_name` from."""
    try:
        from fastembed import TextEmbedding
    except ImportError:
        return None
    for m in TextEmbedding.list_supported_models():
        if m.get("model") == model_name:
            src = m.get("sources") or {}
            return src.get("hf")
    return None


def _cmd_download(model_name: str, cache_dir: str, attempts: int) -> int:
    """Resumable model fetch for flaky/slow networks. huggingface_hub resumes
    its `.incomplete` files across attempts; a short per-attempt read timeout
    breaks the stalls we see on this link, and each retry picks up where the
    last left off."""
    import os
    import time

    repo = _hf_repo_for(model_name)
    if repo is None:
        print(f"could not resolve HF repo for {model_name} (skills extra missing?)")
        return 1
    # Short timeout so a stalled read raises and we resume, instead of hanging.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from huggingface_hub import snapshot_download

    print(f"downloading {repo} -> {cache_dir} (resumable, up to {attempts} tries)")
    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(repo_id=repo, cache_dir=cache_dir, max_workers=2)
            print(f"done: {path}")
            return 0
        except Exception as exc:  # noqa: BLE001 — retry on any network failure
            print(f"  attempt {attempt}/{attempts} interrupted ({type(exc).__name__}); "
                  "resuming…")
            time.sleep(2)
    print("gave up — re-run `jarvis-cli skills download` to keep resuming")
    return 1


def _print_status() -> int:
    recs = scan_skills()
    by_source: dict[str, int] = {}
    for r in recs:
        tag = f"{r.source_tool}" + (f"/{r.plugin}" if r.plugin else "")
        by_source[tag] = by_source.get(tag, 0) + 1
    print(f"discovered {len(recs)} skills:")
    for tag in sorted(by_source):
        print(f"  {by_source[tag]:3d}  {tag}")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    if args.skills_cmd == "status":
        return _print_status()

    cfg = load_config(DEFAULT_CONFIG_PATH).skills
    if args.skills_cmd == "download":
        return _cmd_download(cfg.model_name, cfg.cache_dir, args.attempts)

    try:
        from .embedder import Embedder
        from .index import ensure_index
        from .retriever import SkillRetriever
    except ImportError:
        print("skills extra not installed — run: uv sync --extra skills")
        return 1

    records = scan_skills()
    embedder = Embedder(cfg.model_name, cache_dir=cfg.cache_dir)
    try:
        index = ensure_index(Path(cfg.index_dir), embedder, records)
    except Exception as exc:  # noqa: BLE001 — surface the real failure to the user
        print(f"index build failed: {exc}")
        return 1

    if args.skills_cmd == "index":
        print(f"index ready: {len(index.records)} skills, model={index.model_name}")
        return 0

    # query
    retriever = SkillRetriever(embedder, index)
    text = " ".join(args.text)
    matches = retriever.query(text, k=cfg.top_k)
    print(f"query: {text!r}")
    for m in matches:
        tier = (
            "BODY" if m.score >= cfg.high_threshold
            else "menu" if m.score >= cfg.med_threshold
            else "—"
        )
        print(f"  {m.score:.3f} [{tier:4s}] {m.record.name}")
    return 0
