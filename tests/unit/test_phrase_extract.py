from jarvis_cc.phrase.extract import extract


def test_extract_empty_input_returns_empty():
    assert extract("Bash", {}) == ""
    assert extract(None, {}) == ""
    assert extract("Bash", None) == ""  # type: ignore[arg-type]


def test_extract_bash_returns_command_truncated():
    out = extract("Bash", {"command": "rm -rf /tmp/foo"})
    assert out == "rm -rf /tmp/foo"


def test_extract_bash_truncates_long_command():
    long_cmd = "echo " + "x" * 500
    out = extract("Bash", {"command": long_cmd})
    assert len(out) <= 200


def test_extract_write_uses_basename():
    out = extract("Write", {"file_path": "/Users/jobin/proj/config.toml", "content": "..."})
    assert out == "write config.toml"


def test_extract_edit_uses_basename():
    out = extract("Edit", {"file_path": "/a/b/foo.py", "old_string": "x", "new_string": "y"})
    assert out == "edit foo.py"


def test_extract_multiedit_uses_basename():
    out = extract("MultiEdit", {"file_path": "/a/b/foo.py"})
    assert out == "edit foo.py"


def test_extract_read_uses_basename():
    assert extract("Read", {"file_path": "/a/b/c.md"}) == "read c.md"


def test_extract_grep_quotes_pattern():
    out = extract("Grep", {"pattern": "def main"})
    assert out == "grep 'def main'"


def test_extract_grep_truncates_long_pattern():
    out = extract("Grep", {"pattern": "x" * 200})
    assert len(out) <= 200


def test_extract_glob_quotes_pattern():
    out = extract("Glob", {"pattern": "**/*.py"})
    assert out == "glob '**/*.py'"


def test_extract_webfetch_includes_url():
    out = extract("WebFetch", {"url": "https://example.com/secret", "prompt": "x"})
    assert out.startswith("fetch https://example.com")


def test_extract_websearch_uses_query():
    out = extract("WebSearch", {"query": "site:example.com baz"})
    assert out.startswith("search")
    assert "baz" in out


def test_extract_websearch_without_query_falls_back():
    assert extract("WebSearch", {"other": "x"}) == "search"


def test_extract_unknown_tool_dumps_json():
    out = extract("SomeNewTool", {"foo": "bar", "n": 1})
    assert '"foo"' in out
    assert '"bar"' in out
    assert len(out) <= 200


def test_extract_write_without_file_path_falls_back():
    assert extract("Write", {"content": "x"}) == "write"
