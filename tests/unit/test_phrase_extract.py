from jarvis.phrase.extract import extract, extract_failure


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


def test_extract_bash_with_none_command_returns_empty():
    assert extract("Bash", {"command": None}) == ""


def test_extract_bash_with_empty_command_returns_empty():
    assert extract("Bash", {"command": ""}) == ""


def test_extract_grep_with_none_pattern_falls_back():
    assert extract("Grep", {"pattern": None}) == "grep"


def test_extract_glob_with_none_pattern_falls_back():
    assert extract("Glob", {"pattern": None}) == "glob"


def test_extract_webfetch_with_none_url_falls_back():
    assert extract("WebFetch", {"url": None}) == "fetch"


def test_extract_websearch_with_none_query_falls_back():
    assert extract("WebSearch", {"query": None}) == "search"


def test_extract_unknown_tool_dumps_json():
    out = extract("SomeNewTool", {"foo": "bar", "n": 1})
    assert '"foo"' in out
    assert '"bar"' in out
    assert len(out) <= 200


def test_extract_write_without_file_path_falls_back():
    assert extract("Write", {"content": "x"}) == "write"


def test_extract_askuserquestion_returns_question_plus_options():
    out = extract(
        "AskUserQuestion",
        {
            "questions": [
                {
                    "question": "Pick a colour",
                    "options": [{"label": "Red"}, {"label": "Blue"}],
                }
            ]
        },
    )
    assert "Pick a colour" in out
    assert "Red" in out and "Blue" in out
    # Structured so the LLM sees clear question/options split.
    assert "ask:" in out and "options:" in out


def test_extract_askuserquestion_keeps_cjk_content():
    out = extract(
        "AskUserQuestion",
        {
            "questions": [
                {
                    "question": "你想对博客做哪方面的调整",
                    "options": [{"label": "新增博客文章"}, {"label": "调整主题样式"}],
                }
            ]
        },
    )
    assert "你想对博客做哪方面的调整" in out
    assert "新增博客文章" in out
    assert "调整主题样式" in out


def test_extract_askuserquestion_marks_extra_questions(tmp_path=None):
    out = extract(
        "AskUserQuestion",
        {
            "questions": [
                {"question": "Q1", "options": [{"label": "A"}]},
                {"question": "Q2", "options": [{"label": "B"}]},
                {"question": "Q3", "options": [{"label": "C"}]},
            ]
        },
    )
    assert "Q1" in out
    # Only the first question is unfolded into options; the rest are counted.
    assert "Q2" not in out
    assert "+2 more" in out


def test_extract_askuserquestion_empty_questions_returns_empty():
    assert extract("AskUserQuestion", {"questions": []}) == ""
    assert extract("AskUserQuestion", {}) == ""


def test_extract_askuserquestion_truncates_overlong_fields():
    out = extract(
        "AskUserQuestion",
        {
            "questions": [
                {
                    "question": "x" * 300,
                    "options": [{"label": "y" * 300}],
                }
            ]
        },
    )
    # Each field is capped so the prompt stays bounded.
    assert len(out) <= 400


# --- extract_failure (tool_failure events) ----------------------------------


def test_extract_failure_combines_tool_action_and_error_gist():
    out = extract_failure(
        "Bash",
        {"command": "npm test", "tool_response": {"error": "3 tests failed"}},
    )
    assert "Bash failed" in out
    assert "npm test" in out
    assert "3 tests failed" in out


def test_extract_failure_accepts_string_tool_response():
    out = extract_failure(
        "Bash", {"command": "make", "tool_response": "exit code 2"}
    )
    assert "Bash failed" in out
    assert "exit code 2" in out


def test_extract_failure_falls_back_to_stderr_then_stdout():
    out = extract_failure("Bash", {"tool_response": {"stderr": "boom"}})
    assert "boom" in out
    out2 = extract_failure("Bash", {"tool_response": {"stdout": "noisy"}})
    assert "noisy" in out2


def test_extract_failure_without_tool_response_still_names_tool():
    out = extract_failure("Edit", {"file_path": "/a/b/foo.py"})
    assert "Edit failed" in out
    assert "foo.py" in out


def test_extract_failure_caps_long_error():
    out = extract_failure(
        "Bash", {"tool_response": {"error": "x" * 500}}
    )
    assert len(out) <= 200


def test_extract_failure_empty_returns_empty_when_no_tool():
    assert extract_failure(None, {}) == ""
    assert extract_failure(None, None) == ""
