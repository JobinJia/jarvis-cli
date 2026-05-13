from pathlib import Path

import pytest

from jarvis_cc.config import Config, load_config


def test_load_config_returns_defaults_when_file_missing(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.llm.provider == "deepseek"
    assert cfg.llm.fallback == "ollama"
    assert cfg.tts.provider == "xtts"
    assert cfg.behavior.dedup_window_seconds == 10
    assert cfg.behavior.queue_max_size == 5
    assert cfg.behavior.voice_language == "auto"
    assert cfg.behavior.events == [
        "permission_prompt",
        "idle_prompt",
        "elicitation_dialog",
    ]


def test_load_config_reads_toml(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text(
        """
[llm]
provider = "anthropic"
fallback = "deepseek"

[behavior]
dedup_window_seconds = 30
queue_max_size = 9
"""
    )
    cfg = load_config(p)
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.fallback == "deepseek"
    assert cfg.behavior.dedup_window_seconds == 30
    assert cfg.behavior.queue_max_size == 9
    # Untouched fields stay default
    assert cfg.tts.provider == "xtts"


def test_load_config_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg.paths.socket.startswith(str(tmp_path))
