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


def test_forward_event_translates_askuserquestion_pretooluse(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "Pick a colour",
                    "header": "Colour",
                    "multiSelect": False,
                    "options": [{"label": "Red"}, {"label": "Blue"}],
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
    assert line["lang"] == "en"
    assert "Pick a colour" in line["text"]
    assert "Option one" in line["text"] and "Red" in line["text"]
    assert "Option two" in line["text"] and "Blue" in line["text"]


def test_forward_event_default_mode_is_english_even_for_cjk_question(tmp_path: Path):
    """Default lang_mode='en' wraps CJK questions in English scaffolding
    ('Sir, ... Option one: ...') and tags lang=en, regardless of body content."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {
                    "question": "选一个颜色",
                    "options": [{"label": "红色"}, {"label": "蓝色"}],
                }
            ],
        },
        "cwd": "/x",
    }

    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)

    assert line["notification_type"] == "ask_user_question"
    assert line["lang"] == "en"
    assert line["text"].startswith("Sir, ")
    assert "选一个颜色" in line["text"]  # body read verbatim
    assert "Option one: 红色" in line["text"]
    assert "Option two: 蓝色" in line["text"]


def test_forward_event_zh_mode_renders_chinese_scaffolding(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {"question": "选一个颜色", "options": [{"label": "红色"}, {"label": "蓝色"}]}
            ],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path, lang_mode="zh") is True
    line = _recv_one(received)
    assert line["lang"] == "zh"
    assert line["text"].startswith("先生，")
    assert "选项一: 红色" in line["text"]
    assert "选项二: 蓝色" in line["text"]


def test_forward_event_auto_mode_switches_by_question_text(tmp_path: Path):
    # English question → English scaffolding.
    sock_en = tmp_path / "en.sock"
    rcv_en = _start_unix_echo_server(sock_en)
    en_payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [{"question": "Pick a colour", "options": [{"label": "Red"}]}],
        },
    }
    assert forward_event(io.StringIO(json.dumps(en_payload)), sock_en, lang_mode="auto") is True
    en_line = _recv_one(rcv_en)
    assert en_line["lang"] == "en"
    assert en_line["text"].startswith("Sir, ")

    # CJK question → Chinese scaffolding.
    sock_zh = tmp_path / "zh.sock"
    rcv_zh = _start_unix_echo_server(sock_zh)
    zh_payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [{"question": "选个颜色", "options": [{"label": "红"}]}],
        },
    }
    assert forward_event(io.StringIO(json.dumps(zh_payload)), sock_zh, lang_mode="auto") is True
    zh_line = _recv_one(rcv_zh)
    assert zh_line["lang"] == "zh"
    assert zh_line["text"].startswith("先生，")


def test_forward_event_mentions_remaining_questions(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [
                {"question": "Q1", "options": [{"label": "A"}, {"label": "B"}]},
                {"question": "Q2", "options": [{"label": "C"}]},
                {"question": "Q3", "options": [{"label": "D"}]},
            ],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)
    assert "Q1" in line["text"]
    assert "Q2" not in line["text"]
    assert "2 more" in line["text"].lower()


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


def test_forward_event_does_not_double_prefix_sir(tmp_path: Path):
    """If the question already opens with 'Sir,' the renderer must not
    prepend another 'Sir,' — that creates audio like 'Sir, Sir, ...'."""
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
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)
    # Exactly one 'Sir' before the comma+space at the head of text.
    assert line["text"].count("Sir,") == 1, line["text"]


def test_forward_event_renders_english_address_even_for_xiansheng_question(tmp_path: Path):
    """Question already addresses 'sir' in Chinese; hook still adds 'Sir, '
    (English identity wins). User would naturally cut the redundant '先生，'
    from their question text — but if they don't, we accept one duplicated
    address rather than guessing translation equivalents."""
    sock_path = tmp_path / "j.sock"
    received = _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {
            "questions": [{"question": "先生，确认部署？", "options": [{"label": "好"}]}],
        },
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is True
    line = _recv_one(received)
    assert line["text"].startswith("Sir, ")
    assert line["lang"] == "en"


def test_forward_event_drops_askuserquestion_without_questions(tmp_path: Path):
    sock_path = tmp_path / "j.sock"
    _start_unix_echo_server(sock_path)
    payload = {
        "hook_event_name": "PreToolUse",
        "tool_name": "AskUserQuestion",
        "tool_input": {"questions": []},
    }
    assert forward_event(io.StringIO(json.dumps(payload)), sock_path) is False
