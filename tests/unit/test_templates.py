from jarvis_cli.phrase.templates import render_template
from jarvis_cli.types import Event


def test_permission_prompt_zh():
    ev = Event(notification_type="permission_prompt", tool_name="Bash")
    text = render_template(ev, lang="zh")
    assert "先生" in text
    assert "Bash" in text


def test_permission_prompt_en():
    ev = Event(notification_type="permission_prompt", tool_name="Edit")
    text = render_template(ev, lang="en")
    assert text.startswith("Sir")
    assert "Edit" in text


def test_idle_prompt_zh():
    ev = Event(notification_type="idle_prompt", tool_name=None)
    text = render_template(ev, lang="zh")
    assert "先生" in text


def test_elicitation_dialog_en():
    ev = Event(notification_type="elicitation_dialog", tool_name=None)
    text = render_template(ev, lang="en")
    assert "Sir" in text
