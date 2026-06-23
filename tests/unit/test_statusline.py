"""Tests for the `jarvis-cli-statusline` console script."""
import io
import json
import socket
import threading
from pathlib import Path

import pytest

from jarvis_cli import statusline
from jarvis_cli.config import load_config


def test_fmt_tokens_scales():
    assert statusline._fmt_tokens(500) == "500"
    assert statusline._fmt_tokens(1500) == "1.5k"
    assert statusline._fmt_tokens(2_300_000) == "2.3M"


def test_render_line_full_payload(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(statusline, "_git_branch", lambda cwd: "main")
    data = {
        "model": {"display_name": "Opus"},
        "cwd": "/tmp/proj",
        "cost": {"total_cost_usd": 0.1234},
        "context_window": {
            "total_input_tokens": 15000,
            "total_output_tokens": 500,
            "used_percentage": 8,
        },
    }
    line = statusline.render_line(data)
    assert "[Opus]" in line
    assert "15.5k tok" in line
    assert "8%" in line
    assert "$0.12" in line
    assert "🌿 main" in line


def test_render_line_tolerates_empty_payload(monkeypatch: pytest.MonkeyPatch):
    """Before the first API response CC sends a near-empty payload — must not raise."""
    monkeypatch.setattr(statusline, "_git_branch", lambda cwd: None)
    line = statusline.render_line({})
    assert "[Claude]" in line
    assert "0 tok" in line


def test_render_line_null_cost_and_pct_omitted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(statusline, "_git_branch", lambda cwd: None)
    data = {
        "model": {"display_name": "Sonnet"},
        "cost": {"total_cost_usd": None},
        "context_window": {"used_percentage": None},
    }
    line = statusline.render_line(data)
    assert "💰" not in line
    assert "%" not in line


@pytest.mark.parametrize(
    "cost,step,last,expected",
    [
        (1.5, 2.0, 0.0, None),   # below first step
        (2.0, 2.0, 0.0, 2.0),    # exactly first step
        (3.9, 2.0, 0.0, 2.0),    # first step crossed, not yet second
        (4.1, 2.0, 2.0, 4.0),    # second step crossed
        (4.1, 2.0, 4.0, None),   # already announced 4
        (10.0, 2.0, 2.0, 10.0),  # vault several steps -> announce reached level
    ],
)
def test_crossed_milestone(cost, step, last, expected):
    assert statusline._crossed_milestone(cost, step, last) == expected


def test_milestone_state_roundtrip(tmp_path: Path):
    p = statusline._milestone_state_path(str(tmp_path), "sess-123")
    assert statusline._read_last_milestone(p) == 0.0
    statusline._write_last_milestone(p, 4.0)
    assert statusline._read_last_milestone(p) == 4.0


def test_main_never_crashes_on_garbage_stdin(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json at all"))
    monkeypatch.setattr(statusline, "_git_branch", lambda cwd: None)
    assert statusline.main() == 0
    out = capsys.readouterr().out
    # Garbage stdin -> empty payload -> safe fallback line, never a blank row.
    assert out.startswith("[Claude]")


def test_main_prints_line_and_exits_zero(monkeypatch: pytest.MonkeyPatch, capsys):
    payload = json.dumps(
        {"model": {"display_name": "Opus"}, "context_window": {"total_input_tokens": 1000}}
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(payload))
    monkeypatch.setattr(statusline, "_git_branch", lambda cwd: None)
    # Disabled milestones (default) -> no socket contact.
    monkeypatch.setattr(statusline, "_maybe_announce_cost", lambda data: None)
    assert statusline.main() == 0
    assert "[Opus]" in capsys.readouterr().out


def test_maybe_announce_disabled_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Default config has announce_milestones=False — no daemon contact, no state file."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    monkeypatch.setattr(statusline, "DEFAULT_CONFIG_PATH", str(cfg_path), raising=False)
    monkeypatch.setattr(
        "jarvis_cli.config.DEFAULT_CONFIG_PATH", str(cfg_path), raising=False
    )
    sent: list = []
    monkeypatch.setattr(statusline, "_announce", lambda *a: sent.append(a))
    statusline._maybe_announce_cost({"cost": {"total_cost_usd": 99.0}, "session_id": "s"})
    assert sent == []


def test_maybe_announce_fires_once_per_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[cost]\n"
        "announce_milestones = true\n"
        "milestone_usd = 2.0\n"
        f'state_dir = "{tmp_path / "state"}"\n'
    )
    monkeypatch.setattr(
        "jarvis_cli.config.DEFAULT_CONFIG_PATH", str(cfg_path), raising=False
    )
    monkeypatch.setattr(statusline, "DEFAULT_CONFIG_PATH", str(cfg_path), raising=False)
    sent: list = []
    monkeypatch.setattr(
        statusline, "_announce", lambda text, sock, sid: sent.append(text)
    )

    data = {"cost": {"total_cost_usd": 2.5}, "session_id": "abc"}
    statusline._maybe_announce_cost(data)
    statusline._maybe_announce_cost(data)  # same cost again -> no repeat
    assert len(sent) == 1
    assert "2 dollars" in sent[0]

    # Cross the next threshold -> one more announcement.
    statusline._maybe_announce_cost({"cost": {"total_cost_usd": 4.1}, "session_id": "abc"})
    assert len(sent) == 2
    assert "4 dollars" in sent[1]


def test_announce_swallows_dead_socket():
    """Daemon down -> _announce must not raise (statusline must never stall CC)."""
    statusline._announce("hi", "/nonexistent/jarvis.sock", "s")


def test_announce_reaches_listening_socket():
    """Pre-baked idle_prompt line lands on the daemon socket verbatim."""
    import tempfile

    # AF_UNIX paths are length-capped (~104 bytes on macOS); mkdtemp under the
    # short system tmp root stays well under it where pytest's tmp_path would not.
    sock_dir = tempfile.mkdtemp()
    sock_path = str(Path(sock_dir) / "j.sock")
    received: list[bytes] = []

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def _accept():
        conn, _ = srv.accept()
        received.append(conn.recv(4096))
        conn.close()

    t = threading.Thread(target=_accept)
    t.start()
    statusline._announce("Sir, this session has passed two dollars.", sock_path, "s")
    t.join(timeout=2)
    srv.close()

    assert received, "daemon socket received nothing"
    payload = json.loads(received[0].decode("utf-8").strip())
    assert payload["notification_type"] == "idle_prompt"
    assert payload["text"].endswith("dollars.")
    assert payload["session_id"] is None


def test_cost_config_defaults_and_override(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.cost.announce_milestones is False
    assert cfg.cost.milestone_usd == 2.0
    assert cfg.cost.state_dir.endswith("/cost")

    p = tmp_path / "c.toml"
    p.write_text(
        "[cost]\nannounce_milestones = true\nmilestone_usd = 5.0\n"
    )
    cfg = load_config(p)
    assert cfg.cost.announce_milestones is True
    assert cfg.cost.milestone_usd == 5.0
