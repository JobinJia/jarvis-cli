import json
from pathlib import Path

from jarvis_cc.install import (
    PLIST_LABEL,
    merge_claude_settings,
    remove_from_claude_settings,
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
        env={"PATH": "/opt/homebrew/bin"},
    )
    assert "<string>com.jobin.jarvis-cc</string>" in plist
    assert "<string>/usr/local/bin/jarvis-cc-daemon</string>" in plist
    assert "<key>KeepAlive</key>" in plist
    assert str(tmp_path) in plist
    assert "<key>PATH</key>" in plist


def test_merge_settings_registers_userpromptsubmit_and_posttooluse():
    out = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    for hook_type in ("UserPromptSubmit", "PostToolUse"):
        entries = out["hooks"][hook_type]
        assert len(entries) == 1
        assert entries[0]["hooks"][0]["command"] == "jarvis-cc-hook"


def test_merge_settings_idempotent_for_new_hooks():
    out1 = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    out2 = merge_claude_settings(out1, hook_command="jarvis-cc-hook")
    for hook_type in ("UserPromptSubmit", "PostToolUse"):
        entries = out2["hooks"][hook_type]
        count = sum(
            1 for matcher in entries for h in matcher["hooks"]
            if h["command"] == "jarvis-cc-hook"
        )
        assert count == 1


def test_merge_settings_preserves_existing_userpromptsubmit_entries():
    existing = {
        "hooks": {
            "UserPromptSubmit": [
                {"matcher": "", "hooks": [{"type": "command", "command": "other-hook"}]}
            ]
        }
    }
    out = merge_claude_settings(existing, hook_command="jarvis-cc-hook")
    cmds = [
        h["command"]
        for m in out["hooks"]["UserPromptSubmit"]
        for h in m["hooks"]
    ]
    assert "other-hook" in cmds
    assert "jarvis-cc-hook" in cmds


def test_remove_strips_our_userpromptsubmit_and_posttooluse_entries():
    existing = merge_claude_settings({}, hook_command="jarvis-cc-hook")
    out = remove_from_claude_settings(existing, hook_command="jarvis-cc-hook")
    for hook_type in ("UserPromptSubmit", "PostToolUse", "Notification"):
        entries = out.get("hooks", {}).get(hook_type, [])
        cmds = [h["command"] for m in entries for h in m["hooks"]]
        assert "jarvis-cc-hook" not in cmds


def test_render_plist_embeds_env_vars():
    plist = render_plist(
        label=PLIST_LABEL,
        program="/x",
        log_dir="/y",
        env={"PATH": "/z", "DEEPSEEK_API_KEY": "sk-abc"},
    )
    assert "<key>DEEPSEEK_API_KEY</key>" in plist
    assert "<string>sk-abc</string>" in plist


# ---------------------------------------------------------------------------
# Codex config.toml merge
# ---------------------------------------------------------------------------

from jarvis_cc.install import merge_codex_config, remove_from_codex_config


def test_merge_codex_config_into_empty_file():
    out = merge_codex_config("", hook_command="/abs/jarvis-cc-hook")
    assert "# === jarvis-cc:start ===" in out
    assert "# === jarvis-cc:end ===" in out
    assert 'notify = ["/abs/jarvis-cc-hook"]' in out
    assert "[[hooks.PreToolUse]]" in out
    assert "[[hooks.PermissionRequest]]" in out
    assert "[[hooks.UserPromptSubmit]]" in out
    assert "[[hooks.PostToolUse]]" in out


def test_merge_codex_config_inserts_before_first_section_to_keep_toml_valid():
    """The user's existing file has top-level keys followed by sections.
    Our managed block contains both a top-level key (`notify`) and
    sections — it must land between the user's top-level keys and the
    user's first section, or TOML rejects the file (top-level keys may
    not appear after a section header)."""
    existing = (
        'model = "gpt-5.5"\n'
        'model_reasoning_effort = "high"\n'
        "\n"
        '[projects."/x"]\n'
        'trust_level = "trusted"\n'
    )
    out = merge_codex_config(existing, hook_command="/abs/jarvis-cc-hook")

    assert out.index('model = "gpt-5.5"') < out.index("notify = ")
    assert out.index("# === jarvis-cc:start ===") < out.index('[projects."/x"]')
    assert '[projects."/x"]' in out
    assert 'trust_level = "trusted"' in out


def test_merge_codex_config_is_idempotent_replaces_existing_block():
    """Running install twice (e.g. after a `uv sync` that moves the
    hook binary path) must replace our block in place, not duplicate
    it or accumulate stale entries."""
    first = merge_codex_config(
        'model = "x"\n[projects."/x"]\n',
        hook_command="/old/jarvis-cc-hook",
    )
    second = merge_codex_config(first, hook_command="/new/jarvis-cc-hook")

    assert second.count("# === jarvis-cc:start ===") == 1
    assert second.count("# === jarvis-cc:end ===") == 1
    assert "/old/jarvis-cc-hook" not in second
    assert "/new/jarvis-cc-hook" in second


def test_remove_from_codex_config_strips_managed_block_only():
    existing = (
        'model = "gpt-5.5"\n'
        "\n"
        '[projects."/x"]\n'
        'trust_level = "trusted"\n'
    )
    patched = merge_codex_config(existing, hook_command="/abs/jarvis-cc-hook")
    cleaned = remove_from_codex_config(patched)

    assert "# === jarvis-cc:start ===" not in cleaned
    assert "notify = " not in cleaned
    assert "[[hooks." not in cleaned
    assert 'model = "gpt-5.5"' in cleaned
    assert '[projects."/x"]' in cleaned


def test_remove_from_codex_config_is_safe_when_block_absent():
    untouched = 'model = "gpt-5.5"\n[projects."/x"]\n'
    assert remove_from_codex_config(untouched) == untouched


def test_merge_codex_config_no_sections_only_top_level_keys():
    existing = 'model = "gpt-5.5"\n'
    out = merge_codex_config(existing, hook_command="/abs/jarvis-cc-hook")
    assert 'model = "gpt-5.5"' in out
    assert "# === jarvis-cc:start ===" in out
    assert out.index('model = "gpt-5.5"') < out.index("# === jarvis-cc:start ===")


def test_merge_codex_config_output_is_parseable_toml():
    """End-to-end safety: the merged file must round-trip through a
    real TOML parser without errors."""
    import tomllib
    existing = (
        'model = "gpt-5.5"\n'
        "\n"
        '[projects."/x"]\n'
        'trust_level = "trusted"\n'
    )
    out = merge_codex_config(existing, hook_command="/abs/jarvis-cc-hook")
    parsed = tomllib.loads(out)
    assert parsed["model"] == "gpt-5.5"
    assert parsed["notify"] == ["/abs/jarvis-cc-hook"]
    assert len(parsed["hooks"]["PreToolUse"]) == 1
    assert parsed["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "/abs/jarvis-cc-hook"
    assert parsed["projects"]["/x"]["trust_level"] == "trusted"
