"""`jarvis-cli mcp <status|query|register|unregister>` — MCP server registry.

``status`` lists registered servers; ``query`` tests intent matching against
the registry; ``register``/``unregister`` manage entries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import DEFAULT_CONFIG_PATH, load_config
from .registry import McpServerRecord, load_registry, save_registry


def add_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("mcp", help="manage MCP server intent routing")
    ssub = p.add_subparsers(dest="mcp_cmd", required=True)

    ssub.add_parser("status", help="list registered MCP servers")

    q = ssub.add_parser("query", help="test intent matching against the registry")
    q.add_argument("text", nargs="+", help="the prompt text to test")

    reg = ssub.add_parser("register", help="add a server to the registry")
    reg.add_argument("name", help="unique server name")
    reg.add_argument("--description", required=True, help="what the server does")
    reg.add_argument("--keywords", default="", help="comma-separated search keywords")
    reg.add_argument(
        "--connect", required=True,
        help="JSON string with add_server params (type, url/command/args)",
    )

    unreg = ssub.add_parser("unregister", help="remove a server from the registry")
    unreg.add_argument("name", help="server name to remove")

    ssub.add_parser("index", help="(re)build the embedding index")

    p.set_defaults(func=cmd_mcp)


def _print_status(cfg) -> int:
    records = load_registry(cfg.mcp.registry_path)
    if not records:
        print("no MCP servers registered.")
        print(f"  registry: {cfg.mcp.registry_path}")
        return 0
    print(f"{len(records)} registered MCP server(s):")
    for r in records:
        kw = ", ".join(r.keywords[:5]) if r.keywords else ""
        print(f"  {r.name:20s}  {r.description[:60]}")
        if kw:
            print(f"  {'':20s}  keywords: {kw}")
    print(f"\nregistry: {cfg.mcp.registry_path}")
    return 0


def _cmd_register(args: argparse.Namespace, cfg) -> int:
    records = load_registry(cfg.mcp.registry_path)
    if any(r.name == args.name for r in records):
        print(f"server {args.name!r} already registered; unregister first.")
        return 1
    try:
        connect = json.loads(args.connect)
    except json.JSONDecodeError as exc:
        print(f"invalid --connect JSON: {exc}")
        return 1
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    rec = McpServerRecord(
        name=args.name,
        description=args.description,
        keywords=keywords,
        connect=connect,
    )
    records.append(rec)
    save_registry(records, cfg.mcp.registry_path)
    print(f"registered {args.name!r}. Re-run `jarvis-cli mcp index` to rebuild embeddings.")
    return 0


def _cmd_unregister(args: argparse.Namespace, cfg) -> int:
    records = load_registry(cfg.mcp.registry_path)
    before = len(records)
    records = [r for r in records if r.name != args.name]
    if len(records) == before:
        print(f"server {args.name!r} not found in registry.")
        return 1
    save_registry(records, cfg.mcp.registry_path)
    print(f"removed {args.name!r}. Re-run `jarvis-cli mcp index` to rebuild embeddings.")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    cfg = load_config(DEFAULT_CONFIG_PATH)

    if args.mcp_cmd == "status":
        return _print_status(cfg)
    if args.mcp_cmd == "register":
        return _cmd_register(args, cfg)
    if args.mcp_cmd == "unregister":
        return _cmd_unregister(args, cfg)

    try:
        from ..retrieval.embedder import Embedder
        from ..retrieval.index import ensure_index
        from ..retrieval.retriever import Retriever
    except ImportError:
        print("retrieval stack not installed — run: uv sync --extra skills")
        return 1

    from .registry import load_registry, record_from_dict, record_to_dict

    records = load_registry(cfg.mcp.registry_path)
    if not records:
        print("registry is empty — register servers first.")
        return 1

    embedder = Embedder(cfg.skills.model_name, cache_dir=cfg.skills.cache_dir)
    try:
        index = ensure_index(
            Path(cfg.mcp.index_dir), embedder, records,
            record_to_dict, record_from_dict,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"index build failed: {exc}")
        return 1

    if args.mcp_cmd == "index":
        print(f"index ready: {len(index.records)} servers, model={index.model_name}")
        return 0

    # query
    from .service import _has_lexical_signal, _COSINE_SOLO_FLOOR

    retriever = Retriever(embedder, index)
    text = " ".join(args.text)
    matches = retriever.query(text, k=cfg.mcp.top_k)
    print(f"query: {text!r}")
    for m in matches:
        gated = (
            m.score >= cfg.mcp.med_threshold
            and (_has_lexical_signal(m) or m.cosine >= _COSINE_SOLO_FLOOR)
        )
        if not gated:
            tier = "—"
        elif m.score >= cfg.mcp.high_threshold:
            tier = "CONNECT"
        else:
            tier = "suggest"
        boost = m.score - m.cosine
        print(
            f"  {m.score:.3f} (cos={m.cosine:.3f} +lex={boost:.3f})"
            f" [{tier:7s}] {m.record.name}"
        )
    return 0
