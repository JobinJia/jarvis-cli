"""`jarvis events on|off <type>` edits config.toml by line surgery.

The point of the surgery is what it does NOT touch: the comments inside the
events list (they record why entries are there), the other sections' own
`events` lists, and the entries either side of the one being switched.
"""
import pytest

from jarvis.install import _events_block, _set_event_enabled

_CONFIG = """\
[llm]
provider = "ollama"

[behavior]
humor_level = 1
events = [
  "permission_prompt",
  # task_complete fires after every turn — added 2026-08-28 after three
  # days of silence.
  "task_complete",
  # Tier 2 — opt in by uncommenting:
  # "api_error",
  # "session_end",
]
phrase_target_chars = 70

[webhook]
events = [
  "tool_failure",
]
"""


def _write(tmp_path, text=_CONFIG):
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_off_comments_the_entry_out(tmp_path):
    p = _write(tmp_path)
    _set_event_enabled(p, "task_complete", False)
    out = p.read_text()
    assert '  # "task_complete",\n' in out
    assert '\n  "task_complete",\n' not in out


def test_off_keeps_the_lists_own_comments(tmp_path):
    p = _write(tmp_path)
    _set_event_enabled(p, "task_complete", False)
    out = p.read_text()
    assert "added 2026-08-28 after three" in out
    assert '"permission_prompt",' in out


def test_on_uncomments_a_tier_two_entry(tmp_path):
    p = _write(tmp_path)
    _set_event_enabled(p, "api_error", True)
    out = p.read_text()
    assert '  "api_error",\n' in out
    # Its neighbour stays commented out.
    assert '# "session_end",' in out


def test_off_then_on_round_trips(tmp_path):
    p = _write(tmp_path)
    _set_event_enabled(p, "task_complete", False)
    _set_event_enabled(p, "task_complete", True)
    assert p.read_text() == _CONFIG


def test_on_inserts_a_type_that_is_absent(tmp_path):
    p = _write(tmp_path)
    _set_event_enabled(p, "context_overflow", True)
    lines = p.read_text().splitlines()
    assert '  "context_overflow",' in lines
    # Inserted inside the [behavior] list, above its closing bracket.
    assert lines.index('  "context_overflow",') < lines.index("]")


def test_the_webhook_events_list_is_left_alone(tmp_path):
    p = _write(tmp_path)
    _set_event_enabled(p, "tool_failure", True)
    out = p.read_text()
    # tool_failure was absent from [behavior] and gets added there; the
    # [webhook] list — which already had it — is untouched.
    assert out.count('"tool_failure",') == 2
    assert out.split("[webhook]")[1].count('"tool_failure",') == 1


def test_already_in_that_state_is_reported_not_duplicated(tmp_path):
    p = _write(tmp_path)
    before = p.read_text()
    assert "already on" in _set_event_enabled(p, "task_complete", True)
    assert "already off" in _set_event_enabled(p, "api_error", False)
    assert p.read_text() == before


def test_a_one_line_list_is_refused_with_an_explanation(tmp_path):
    p = _write(tmp_path, '[behavior]\nevents = ["idle_prompt", "task_complete"]\n')
    with pytest.raises(ValueError, match="one line"):
        _set_event_enabled(p, "task_complete", False)


def test_a_config_without_behavior_is_refused(tmp_path):
    p = _write(tmp_path, '[llm]\nprovider = "ollama"\n')
    with pytest.raises(ValueError, match=r"\[behavior\]"):
        _set_event_enabled(p, "task_complete", False)


def test_events_block_spans_only_the_behavior_list(tmp_path):
    lines = _CONFIG.splitlines(keepends=True)
    first, last = _events_block(lines)
    body = "".join(lines[first:last])
    assert '"task_complete",' in body
    assert '"tool_failure",' not in body
