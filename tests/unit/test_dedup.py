from jarvis_cli.daemon.dedup import DedupWindow
from jarvis_cli.types import Event


def _ev(t: float, tool: str = "Bash", cwd: str = "/x") -> Event:
    return Event(
        notification_type="permission_prompt",
        tool_name=tool,
        cwd=cwd,
        received_at=t,
    )


def test_first_event_is_not_duplicate():
    w = DedupWindow(window_seconds=10)
    assert w.is_duplicate(_ev(0.0)) is False


def test_same_key_within_window_is_duplicate():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0))
    assert w.is_duplicate(_ev(5.0)) is True


def test_same_key_outside_window_is_not_duplicate():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0))
    assert w.is_duplicate(_ev(10.1)) is False


def test_different_keys_are_independent():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0, tool="Bash"))
    assert w.is_duplicate(_ev(1.0, tool="Edit")) is False
    assert w.is_duplicate(_ev(1.0, tool="Bash", cwd="/y")) is False


def test_is_duplicate_updates_last_seen():
    w = DedupWindow(window_seconds=10)
    w.is_duplicate(_ev(0.0))
    w.is_duplicate(_ev(5.0))  # dup → last_seen slid to 5.0; window now expires at 15.0
    # 14.9 is still inside the slid window → dup; this slides last_seen to 14.9
    assert w.is_duplicate(_ev(14.9)) is True
    # 25.1 - 14.9 = 10.2 > 10 → outside window from the freshly-slid anchor → not dup
    assert w.is_duplicate(_ev(25.1)) is False
