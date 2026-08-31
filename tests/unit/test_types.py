from jarvis.types import Event


def test_dedup_key_combines_cwd_type_tool():
    e = Event(
        notification_type="permission_prompt",
        tool_name="Bash",
        cwd="/x/y",
    )
    assert e.dedup_key() == "/x/y::permission_prompt::Bash"


def test_dedup_key_handles_none_fields():
    e = Event(notification_type="idle_prompt", tool_name=None, cwd=None)
    assert e.dedup_key() == "::idle_prompt::"


def test_dedup_key_differs_by_cwd():
    a = Event(notification_type="permission_prompt", tool_name="Bash", cwd="/a")
    b = Event(notification_type="permission_prompt", tool_name="Bash", cwd="/b")
    assert a.dedup_key() != b.dedup_key()
