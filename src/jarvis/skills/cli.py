"""`jarvis skills <status|index|query>` — operate/debug the skill index.

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

    g = ssub.add_parser(
        "govern",
        help="hide standalone skills + disable skill-plugins (reversible)",
    )
    g.add_argument("--mode", default=None, help="skillOverrides state (default user-invocable-only)")
    g.add_argument("--keep", default="", help="comma-separated skill/plugin names to leave visible")
    g.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    ssub.add_parser("restore", help="reverse `govern` from its manifest")
    ssub.add_parser("govern-status", help="show what governance currently manages")

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
    print("gave up — re-run `jarvis skills download` to keep resuming")
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


def _cmd_govern(args: argparse.Namespace) -> int:
    import json

    from .govern import (
        DEFAULT_MODE,
        GovernPaths,
        apply_governance,
        build_plan,
        discover,
    )

    paths = GovernPaths.default()
    records = discover()
    enabled = json.loads(paths.cc_settings.read_text()).get("enabledPlugins", {}) \
        if paths.cc_settings.exists() else {}
    keep = {k.strip() for k in args.keep.split(",") if k.strip()}
    plan = build_plan(
        records, enabled, keep=keep, mode=args.mode or DEFAULT_MODE,
        cc_agents=paths.cc_agents,
    )
    print(f"plan (mode={plan.mode}):")
    print(f"  hide {len(plan.standalone)} standalone skills: {', '.join(plan.standalone) or '—'}")
    print(f"  disable {len(plan.plugins)} skill-plugins: {', '.join(plan.plugins) or '—'}")
    print(f"  re-home {len(plan.agents)} agents: {', '.join(Path(d).name for _, d in plan.agents) or '—'}")
    if plan.codex_skills:
        print(f"  (codex skills detected: {', '.join(plan.codex_skills)})")
    if args.dry_run:
        print("dry-run: nothing changed.")
        return 0
    if plan.is_empty():
        print("nothing to govern.")
        return 0
    apply_governance(plan, paths)
    print(f"applied. manifest: {paths.manifest}")
    print("note: takes effect in NEW Claude Code sessions. `skills restore` to undo.")
    return 0


def _cmd_restore() -> int:
    from .govern import GovernPaths, restore_governance

    res = restore_governance(GovernPaths.default())
    if not res.get("restored"):
        print(f"nothing to restore ({res.get('reason', 'no manifest')}).")
        return 0
    print(f"restored: re-enabled {len(res['plugins_reenabled'])} plugins, "
          f"removed {len(res['skillOverrides_removed'])} skillOverrides, "
          f"deleted {len(res['agents_removed'])} re-homed agents.")
    print("takes effect in NEW Claude Code sessions.")
    return 0


def _cmd_govern_status() -> int:
    import json

    from .govern import GovernPaths

    paths = GovernPaths.default()
    if not paths.manifest.exists():
        print("governance not applied (no manifest).")
        return 0
    m = json.loads(paths.manifest.read_text())
    print(f"governing (mode={m.get('mode')}):")
    print(f"  {len(m.get('skillOverrides', []))} standalone skills hidden")
    print(f"  {len(m.get('plugins', []))} skill-plugins disabled: {', '.join(m.get('plugins', []))}")
    print(f"  {len(m.get('agents', []))} agents re-homed")
    return 0


def cmd_skills(args: argparse.Namespace) -> int:
    if args.skills_cmd == "status":
        return _print_status()
    if args.skills_cmd == "govern":
        return _cmd_govern(args)
    if args.skills_cmd == "restore":
        return _cmd_restore()
    if args.skills_cmd == "govern-status":
        return _cmd_govern_status()

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
