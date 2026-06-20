import json
from pathlib import Path

from jarvis_cli.mcp.registry import (
    McpServerRecord,
    load_registry,
    save_registry,
)


def _write(path: Path, data: dict | list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


_REGISTRY = {
    "servers": [
        {
            "name": "memex",
            "description": "Session history search",
            "keywords": ["session", "history", "会话"],
            "connect": {"type": "http", "url": "http://localhost:10013/api/mcp"},
        },
        {
            "name": "playwright",
            "description": "Browser automation",
            "keywords": ["browser", "浏览器"],
            "connect": {"type": "stdio", "command": "npx", "args": ["@playwright/mcp"]},
        },
    ]
}


def test_load_registry_parses_servers(tmp_path):
    _write(tmp_path / "registry.json", _REGISTRY)
    records = load_registry(tmp_path / "registry.json")
    assert len(records) == 2
    assert records[0].name == "memex"
    assert records[0].keywords == ["session", "history", "会话"]
    assert records[0].connect["type"] == "http"
    assert records[1].name == "playwright"


def test_load_registry_accepts_bare_list(tmp_path):
    _write(tmp_path / "registry.json", _REGISTRY["servers"])
    records = load_registry(tmp_path / "registry.json")
    assert len(records) == 2


def test_load_registry_skips_nameless_entries(tmp_path):
    data = {"servers": [{"description": "no name"}, {"name": "ok", "description": "d"}]}
    _write(tmp_path / "registry.json", data)
    records = load_registry(tmp_path / "registry.json")
    assert len(records) == 1
    assert records[0].name == "ok"


def test_load_missing_registry_returns_empty(tmp_path):
    records = load_registry(tmp_path / "nonexistent.json")
    assert records == []


def test_save_and_reload_roundtrip(tmp_path):
    original = [
        McpServerRecord(
            name="test-server",
            description="A test MCP server",
            keywords=["test", "测试"],
            connect={"type": "http", "url": "http://localhost:8080"},
        ),
    ]
    path = tmp_path / "registry.json"
    save_registry(original, path)
    loaded = load_registry(path)
    assert len(loaded) == 1
    assert loaded[0].name == "test-server"
    assert loaded[0].description == "A test MCP server"
    assert loaded[0].keywords == ["test", "测试"]
    assert loaded[0].connect == {"type": "http", "url": "http://localhost:8080"}


def test_text_for_embedding_includes_name_and_description():
    rec = McpServerRecord(
        name="my-server",
        description="Does amazing things",
        keywords=["magic", "wizard"],
    )
    text = rec.text_for_embedding()
    assert "my server" in text  # deslug
    assert "Does amazing things" in text
    assert "magic" in text


def test_content_hash_changes_with_description():
    r1 = load_registry.__wrapped__ if hasattr(load_registry, "__wrapped__") else None  # noqa
    rec_a = McpServerRecord(name="s", description="alpha", keywords=[])
    rec_b = McpServerRecord(name="s", description="beta", keywords=[])
    from jarvis_cli.mcp.registry import _compute_hash

    assert _compute_hash(rec_a) != _compute_hash(rec_b)
