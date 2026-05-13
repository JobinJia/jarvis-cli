from jarvis_cc.phrase.prompt import build_messages
from jarvis_cc.types import Event


def test_build_messages_zh_includes_jarvis_system_and_event():
    ev = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        tool_input={"command": "rm foo"},
        cwd="/x",
    )
    msgs = build_messages(ev, lang="zh", max_chars=30)
    assert msgs[0]["role"] == "system"
    assert "J.A.R.V.I.S" in msgs[0]["content"] or "管家" in msgs[0]["content"]
    assert "中文" in msgs[0]["content"]
    assert "30" in msgs[0]["content"]
    assert msgs[-1]["role"] == "user"
    assert "permission_prompt" in msgs[-1]["content"]
    assert "Bash" in msgs[-1]["content"]


def test_build_messages_en_swaps_language_clause():
    ev = Event(notification_type="idle_prompt", tool_name=None)
    msgs = build_messages(ev, lang="en", max_chars=30)
    assert "English" in msgs[0]["content"]
