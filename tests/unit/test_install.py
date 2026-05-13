import json
from pathlib import Path

from jarvis_cc.install import (
    PLIST_LABEL,
    merge_claude_settings,
    render_plist,
)


def test_merge_settings_into_empty():
    out = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    assert out["hooks"]["Notification"][0]["hooks"][0]["command"] == "jarvis-cc-hook"


def test_merge_settings_preserves_other_hooks():
    existing = {
        "hooks": {
            "PreToolUse": [{"matcher": "", "hooks": [{"type": "command", "command": "x"}]}]
        }
    }
    out = merge_claude_settings(existing, hook_command="jarvis-cc-hook")
    assert "PreToolUse" in out["hooks"]
    assert "Notification" in out["hooks"]


def test_merge_settings_is_idempotent():
    out1 = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    out2 = merge_claude_settings(out1, hook_command="jarvis-cc-hook")
    notif = out2["hooks"]["Notification"]
    # Should not duplicate our entry
    assert sum(1 for n in notif for h in n["hooks"] if h["command"] == "jarvis-cc-hook") == 1


def test_render_plist_contains_label_and_program(tmp_path: Path):
    plist = render_plist(
        label=PLIST_LABEL,
        program="/usr/local/bin/jarvis-cc-daemon",
        log_dir=str(tmp_path),
    )
    assert "<string>com.jobin.jarvis-cc</string>" in plist
    assert "<string>/usr/local/bin/jarvis-cc-daemon</string>" in plist
    assert "<key>KeepAlive</key>" in plist
    assert str(tmp_path) in plist
