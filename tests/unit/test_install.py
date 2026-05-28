import json
import tomllib
from pathlib import Path

import httpx
import pytest

from jarvis_cc.install import (
    PLIST_LABEL,
    EnvScan,
    WizardChoices,
    _render_configured_toml,
    merge_claude_settings,
    recommend_profile,
    remove_from_claude_settings,
    render_plist,
    scan_environment,
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


# ---------------------------------------------------------------------------
# Install wizard
# ---------------------------------------------------------------------------


def test_recommend_profile_prefers_local_when_ollama_and_cosyvoice_present():
    env = EnvScan(ollama_up=True, has_cosyvoice_model=True, has_deepseek_key=True)
    assert recommend_profile(env) == "local-zero-cost"


def test_recommend_profile_picks_cloud_cheap_when_ollama_missing_but_keys_present():
    env = EnvScan(has_deepseek_key=True, has_elevenlabs_key=True)
    assert recommend_profile(env) == "cloud-cheap"


def test_recommend_profile_falls_back_to_say_only_when_nothing_usable():
    """Operator with no LLM keys, no Ollama, no voice clone still must
    get *a* working profile so they hear something on first install."""
    assert recommend_profile(EnvScan()) == "say-only"


def test_recommend_profile_skips_cloud_cheap_without_elevenlabs():
    """A cloud LLM key alone isn't enough for the cloud-cheap profile —
    that path requires both LLM and ElevenLabs for the voice."""
    env = EnvScan(has_deepseek_key=True)
    assert recommend_profile(env) == "say-only"


def test_scan_environment_reads_keys_from_env_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Key detection must be pure env reads — no shelling out, no network."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    # Stub the Ollama probe so the test doesn't depend on a live daemon.
    def _no_ollama(*a, **kw):
        raise httpx.ConnectError("no daemon in test")

    monkeypatch.setattr(httpx, "get", _no_ollama)
    env = scan_environment(tmp_path)
    assert env.has_deepseek_key is True
    assert env.has_anthropic_key is False
    assert env.has_openai_key is False
    assert env.has_elevenlabs_key is False
    assert env.ollama_up is False


def test_scan_environment_detects_voice_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "jarvis_en.wav").write_bytes(b"RIFF...")
    (tmp_path / "models" / "cosyvoice3-0.5b-candle").mkdir(parents=True)
    (tmp_path / "models" / "cosyvoice3-0.5b-candle" / "model.safetensors").write_bytes(b"\x00")
    monkeypatch.setattr(
        httpx, "get",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("nope")),
    )
    env = scan_environment(tmp_path)
    assert env.has_jarvis_en_voice is True
    assert env.has_cosyvoice_model is True
    assert env.has_xtts_model is False


def test_scan_environment_detects_live_ollama(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When Ollama is reachable, we record its name and pulled models so
    the wizard can show a useful summary instead of a bare ✓."""

    class _R:
        status_code = 200
        def json(self):
            return {"models": [{"name": "qwen3:8b"}, {"name": "llama3:8b"}]}

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: _R())
    env = scan_environment(tmp_path)
    assert env.ollama_up is True
    assert env.ollama_models == ["qwen3:8b", "llama3:8b"]


def test_render_configured_toml_round_trips_through_load_config(tmp_path: Path):
    """The wizard's output must parse cleanly via load_config — anything
    else means a fresh-install user gets a broken daemon."""
    from jarvis_cc.config import load_config

    choices = WizardChoices(
        humor_level=2, voice_language="en", city="Shanghai",
        profile="local-zero-cost",
    )
    toml = _render_configured_toml(choices, preserve={"ref_text_en": "hello there"})

    # Sanity on the rendered text.
    parsed = tomllib.loads(toml)
    assert parsed["behavior"]["humor_level"] == 2
    assert parsed["behavior"]["voice_language"] == "en"
    assert parsed["behavior"]["session_briefing"]["city"] == "Shanghai"
    assert parsed["llm"]["provider"] == "ollama"
    assert parsed["tts"]["provider"] == "cosyvoice"
    assert parsed["tts"]["cosyvoice"]["ref_text_en"] == "hello there"

    # And via the actual loader the daemon uses.
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml)
    cfg = load_config(cfg_path)
    assert cfg.behavior.humor_level == 2
    assert cfg.behavior.session_briefing.city == "Shanghai"


def test_render_configured_toml_clamps_implicit_humor_within_range(tmp_path: Path):
    """Wizard returns 0-3, but if a config field drift happened we'd want
    the loader's clamp to still bring us back into range. Belt + braces."""
    from jarvis_cc.config import load_config

    choices = WizardChoices(
        humor_level=3, voice_language="auto", city="",
        profile="say-only",
    )
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(_render_configured_toml(choices))
    cfg = load_config(cfg_path)
    assert cfg.behavior.humor_level == 3
    assert cfg.behavior.voice_language == "auto"


def test_render_configured_toml_quotes_ref_text_en_with_apostrophes():
    """Free-form preserved strings (transcripts) often contain quotes —
    they must be JSON-encoded so the rendered TOML is parseable."""
    choices = WizardChoices(
        humor_level=1, voice_language="en", city="",
        profile="local-zero-cost",
    )
    toml = _render_configured_toml(
        choices,
        preserve={"ref_text_en": "I'm \"Jarvis\", at your service."},
    )
    parsed = tomllib.loads(toml)
    assert parsed["tts"]["cosyvoice"]["ref_text_en"] == 'I\'m "Jarvis", at your service.'
