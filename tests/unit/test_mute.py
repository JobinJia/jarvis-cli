"""Mute state: duration parsing, expiry, and what each scope silences."""
import json
import time

import pytest

from jarvis import mute


# --- durations -------------------------------------------------------------


@pytest.mark.parametrize(
    "text,seconds",
    [
        ("60s", 60),
        ("30m", 1800),
        ("1h", 3600),
        ("7d", 604800),
        ("1.5h", 5400),
        ("2H", 7200),
        (" 45m ", 2700),
        ("30", 1800),  # bare number = minutes
    ],
)
def test_parse_duration(text, seconds):
    assert mute.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "soon", "30x", "-5m", "0h", "m30"])
def test_parse_duration_rejects_junk(text):
    with pytest.raises(ValueError):
        mute.parse_duration(text)


def test_format_duration_stops_at_two_units():
    assert mute.format_duration(5400) == "1h30m"
    assert mute.format_duration(604800) == "7d"
    assert mute.format_duration(0) == "0s"


# --- state -----------------------------------------------------------------


def _path(tmp_path):
    return tmp_path / "mute.json"


def test_nothing_is_muted_by_default(tmp_path):
    state = mute.load(_path(tmp_path))
    assert mute.muted_reason(state, "task_complete") is None
    assert mute.describe(state) == {}


def test_global_mute_silences_every_type(tmp_path):
    p = _path(tmp_path)
    mute.set_global(p, 600)
    state = mute.load(p)
    for ntype in ("task_complete", "tool_failure", "session_start"):
        assert mute.muted_reason(state, ntype) is not None
    assert "all" in mute.describe(state)


def test_type_mute_silences_only_that_type(tmp_path):
    p = _path(tmp_path)
    mute.set_type(p, "task_complete", 600)
    state = mute.load(p)
    assert mute.muted_reason(state, "task_complete") is not None
    assert mute.muted_reason(state, "tool_failure") is None


def test_mute_lapses_on_its_own(tmp_path):
    p = _path(tmp_path)
    mute.set_global(p, 600)
    state = mute.load(p)
    # An hour later the same state file no longer silences anything — the
    # expiry is what ends a mute, not a second command.
    later = time.time() + 3600
    assert mute.muted_reason(state, "idle_prompt", now=later) is None
    assert mute.describe(state, now=later) == {}


def test_forever_never_lapses(tmp_path):
    p = _path(tmp_path)
    mute.set_global(p, None)
    state = mute.load(p)
    assert state["global"] == mute.FOREVER
    far = time.time() + 86400 * 365
    assert mute.muted_reason(state, "idle_prompt", now=far) is not None
    assert mute.describe(state)["all"] == "forever"


def test_save_prunes_lapsed_entries(tmp_path):
    p = _path(tmp_path)
    mute.set_type(p, "task_complete", 600)
    mute.save(p, {"global": time.time() - 1, "types": {"tool_failure": 1.0}})
    on_disk = json.loads(p.read_text())
    assert on_disk == {}


def test_unmute_lifts_both_scopes(tmp_path):
    p = _path(tmp_path)
    mute.set_global(p, 600)
    mute.set_type(p, "task_complete", 600)
    assert mute.clear_all(p) is True
    assert mute.describe(mute.load(p)) == {}
    # Nothing left to lift the second time.
    assert mute.clear_all(p) is False


def test_clear_type_leaves_the_others(tmp_path):
    p = _path(tmp_path)
    mute.set_type(p, "task_complete", 600)
    mute.set_type(p, "idle_prompt", 600)
    assert mute.clear_type(p, "task_complete") is True
    state = mute.load(p)
    assert mute.muted_reason(state, "task_complete") is None
    assert mute.muted_reason(state, "idle_prompt") is not None


def test_unreadable_state_never_silences_the_daemon(tmp_path):
    # A half-written or hand-mangled file must fail open: a daemon that
    # cannot parse its mute file has not been asked to stay quiet.
    p = _path(tmp_path)
    p.write_text("{not json at all")
    assert mute.muted_reason(mute.load(p), "task_complete") is None
    p.write_text('["a", "list"]')
    assert mute.muted_reason(mute.load(p), "task_complete") is None
