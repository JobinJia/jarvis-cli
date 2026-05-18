import io
import json
import socket
import threading
from pathlib import Path

from jarvis_cc.hook_client import forward_event


def _start_unix_echo_server(path: Path) -> list[bytes]:
    received: list[bytes] = []
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    sock.listen(1)

    def _serve():
        conn, _ = sock.accept()
        received.append(conn.recv(4096))
        conn.close()
        sock.close()

    threading.Thread(target=_serve, daemon=True).start()
    return received


def test_forward_event_writes_ndjson_to_socket(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)

    payload = {
        "session_id": "abc",
        "notification_type": "permission_prompt",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)

    assert ok is True
    # Allow the server thread to write `received`
    for _ in range(50):
        if received:
            break
        import time
        time.sleep(0.01)
    assert len(received) == 1
    line = received[0].decode().strip()
    assert json.loads(line)["tool_name"] == "Bash"


def test_forward_event_returns_false_when_socket_missing(tmp_path: Path):
    sock_path = tmp_path / "does-not-exist.sock"
    ok = forward_event(io.StringIO('{"notification_type":"idle_prompt"}'), sock_path)
    assert ok is False


def test_forward_event_handles_invalid_json(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    ok = forward_event(io.StringIO("not-json"), sock_path)
    assert ok is False


def test_forward_event_returns_false_for_non_dict_json(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    # Valid JSON, but a list — would have crashed on payload["_received_at"]=...
    ok = forward_event(io.StringIO("[1,2,3]"), sock_path)
    assert ok is False


def _recv_one(received: list[bytes]) -> dict:
    import time as _t

    for _ in range(50):
        if received:
            break
        _t.sleep(0.01)
    assert received, "server thread received nothing"
    return json.loads(received[0].decode().strip())


def test_forward_event_userpromptsubmit_sends_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "abc-123",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row["command"] == "cancel"
    assert row["session_id"] == "abc-123"


def test_forward_event_posttooluse_sends_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "sess-9",
        "cwd": "/x",
    }
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is True
    row = _recv_one(received)
    assert row["command"] == "cancel"
    assert row["session_id"] == "sess-9"


def test_forward_event_userpromptsubmit_without_session_id_is_dropped(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {"hook_event_name": "UserPromptSubmit", "cwd": "/x"}
    ok = forward_event(io.StringIO(json.dumps(payload)), sock_path)
    assert ok is False


def test_forward_event_cancel_disabled_by_flag(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "session_id": "abc",
    }
    ok = forward_event(
        io.StringIO(json.dumps(payload)),
        sock_path,
        cancel_on_user_action=False,
    )
    assert ok is False


def test_forward_event_en_mode_forwards_raw_for_llm_translation(tmp_path: Path):
    """Default `en` mode does NOT pre-bake text; it forwards the question
    payload so the daemon's phrase router calls the LLM to translate/rephrase
    into Jarvis-toned English."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "你想对博客做哪方面的调整",
                    "options": [{"label": "新增博客文章"}, {"label": "调整主题样式"}],
                }
            ],
        },
        "cwd": "/x",
        "session_id": "s1",
    }

    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)

    assert line["notification_type"] == "ask_user_question"
    assert line["tool_name"] == "AskUserQuestion"
    assert line["cwd"] == "/x"
    assert line["session_id"] == "s1"
    assert "text" not in line  # daemon must call router, not synthesize verbatim
    assert "lang" not in line  # daemon picks lang from voice_language config
    # tool_input preserved so the daemon's extract() can summarize it.
    assert line["tool_input"]["questions"][0]["question"] == "你想对博客做哪方面的调整"


def test_forward_event_zh_mode_forwards_raw_for_llm_translation(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [{"question": "Pick a colour", "options": [{"label": "Red"}]}],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="zh") is True
    line = _recv_one(received)
    assert line["notification_type"] == "ask_user_question"
    assert "text" not in line
    assert "lang" not in line


def test_forward_event_auto_mode_renders_verbatim_for_cjk(tmp_path: Path):
    """`auto` mode keeps the no-LLM shortcut: CJK body → Chinese scaffolding
    + verbatim text, English body → English scaffolding + verbatim text."""
    sock_zh = tmp_path / "zh.sock"
    rcv_zh = _start_unix_echo_server(sock_zh)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [{"question": "选个颜色", "options": [{"label": "红"}]}],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_zh, lang_mode="auto") is True
    line = _recv_one(rcv_zh)
    assert line["lang"] == "zh"
    assert line["text"].startswith("先生，")
    assert "选项一: 红" in line["text"]


def test_forward_event_auto_mode_renders_verbatim_for_english(tmp_path: Path):
    sock_en = tmp_path / "en.sock"
    rcv_en = _start_unix_echo_server(sock_en)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "Pick a colour",
                    "options": [{"label": "Red"}, {"label": "Blue"}],
                }
            ],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_en, lang_mode="auto") is True
    line = _recv_one(rcv_en)
    assert line["lang"] == "en"
    assert line["text"].startswith("Sir, ")
    assert "Option one: Red" in line["text"]


def test_forward_event_auto_mode_does_not_double_prefix_sir(tmp_path: Path):
    """Hook scaffolding only applies in auto mode; the no-double-Sir rule
    only matters there."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "Sir, three threads remain. Which first?",
                    "options": [{"label": "A"}],
                }
            ],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="auto") is True
    line = _recv_one(received)
    assert line["text"].count("Sir,") == 1, line["text"]


def test_forward_event_leaves_non_askuserquestion_payloads_unchanged(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "notification_type": "permission_prompt",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "cwd": "/x",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)
    assert line["notification_type"] == "permission_prompt"
    assert line["tool_name"] == "Bash"
    assert "text" not in line


def test_forward_event_drops_askuserquestion_without_questions(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {"questions": []},
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is False


# ---------------------------------------------------------------------------
# Codex CLI integration
# ---------------------------------------------------------------------------
# Codex's lifecycle-hook payloads share Claude Code's snake_case shape and
# hook_event_name vocabulary for the events both products have in common
# (UserPromptSubmit, PostToolUse). Codex adds PermissionRequest, which has
# no Claude Code analogue and so needs explicit translation. The notify
# command uses a completely different flat payload with kebab-case keys.


def test_forward_event_codex_permission_request_translates_to_permission_prompt(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PermissionRequest",
        "session_id": "sess-xyz",
        "turn_id": "turn-1",
        "cwd": "/Users/me/repo",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_use_id": "tu-1",
        "tool_input": {"command": "rm -rf /tmp/old"},
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "permission_prompt"
    assert row["tool_name"] == "Bash"
    assert row["tool_input"] == {"command": "rm -rf /tmp/old"}
    assert row["cwd"] == "/Users/me/repo"
    assert row["session_id"] == "sess-xyz"


def test_forward_event_codex_notify_agent_turn_complete_becomes_idle_prompt(tmp_path: Path):
    """Codex `notify = [...]` sends a flat kebab-case payload when a turn
    finishes — equivalent to Claude Code's idle_prompt notification."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "type": "agent-turn-complete",
        "thread-id": "thr-1",
        "turn-id": "turn-9",
        "cwd": "/Users/me/repo",
        "input-messages": ["please refactor"],
        "last-assistant-message": "Done.",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row["notification_type"] == "idle_prompt"
    assert row["session_id"] == "thr-1"
    assert row["cwd"] == "/Users/me/repo"


def test_forward_event_codex_user_prompt_submit_emits_cancel(tmp_path: Path):
    """Codex's UserPromptSubmit shares CC's name/shape, so cancel behavior
    must work for both without a separate code path."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "sess-codex",
        "cwd": "/x",
        "prompt": "do the thing",
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row.get("command") == "cancel"
    assert row.get("session_id") == "sess-codex"


def test_forward_event_codex_post_tool_use_emits_cancel(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "sess-codex",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "tool_response": {"stdout": "..."},
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    row = _recv_one(received)
    assert row.get("command") == "cancel"


def test_forward_event_codex_session_start_is_dropped(tmp_path: Path):
    """SessionStart is informational only — neither product wants Jarvis
    to speak on every session bring-up. Letting the payload through to
    the daemon with no notification_type would have it filtered there,
    but it's cleaner to drop at the hook."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "sess",
        "source": "startup",
        "cwd": "/x",
    }
    # Pass-through behavior: daemon filters on notification_type.
    # Either drop here (False) or forward as-is and let the daemon ignore.
    # We accept either, just assert it doesn't raise.
    forward_event(io.StringIO(json.dumps(payload)), sock_path)


def test_forward_event_codex_permission_request_with_cancel_disabled(tmp_path: Path):
    """cancel_on_user_action only governs UserPromptSubmit/PostToolUse —
    PermissionRequest is a different axis and must still come through
    even with cancel_on_user_action=False."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PermissionRequest",
        "session_id": "sess",
        "tool_name": "Bash",
        "tool_input": {"command": "ls"},
        "cwd": "/x",
    }
    assert forward_event(
        io.StringIO(json.dumps(payload)), sock_path,
        cancel_on_user_action=False,
    ) is True
    row = _recv_one(received)
    assert row["notification_type"] == "permission_prompt"
