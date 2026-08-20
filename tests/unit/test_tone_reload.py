"""Live tone adjustment: daemon reload_behavior + the tone CLI's config edit."""
from __future__ import annotations

import argparse
import re

import pytest

from jarvis_cli.config import Config
from jarvis_cli.daemon.main import Daemon


@pytest.mark.asyncio
async def test_reload_behavior_swaps_behavior_in_place(tmp_path):
    """The router reads cfg.behavior.* per call off the SAME Config object the
    daemon holds, so replacing `.behavior` must be enough — no reconstruction."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[behavior]\nhumor_level = 3\n", encoding="utf-8")

    d = Daemon(Config(), config_path=cfg_file)
    router_view = d.router.cfg  # same object by reference
    assert d.cfg.behavior.humor_level == 1  # dataclass default

    reply = await d._on_query({"command": "reload_behavior"})

    assert reply == {"ok": True, "humor_level": 3}
    assert router_view.behavior.humor_level == 3


@pytest.mark.asyncio
async def test_reload_behavior_pushes_weather_knobs_into_the_cache(tmp_path):
    """The weather cache holds copies of its two knobs, so swapping
    `.behavior` alone would leave them frozen at daemon-start values while
    every neighbouring [behavior] field reloaded live."""
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "[behavior.session_briefing]\n"
        "weather_ttl_seconds = 900\n"
        "weather_stale_max_age_seconds = 1800\n",
        encoding="utf-8",
    )

    d = Daemon(Config(), config_path=cfg_file)
    assert d._weather_cache.ttl == 600           # dataclass defaults
    assert d._weather_cache.stale_max_age == 7200

    reply = await d._on_query({"command": "reload_behavior"})

    assert reply["ok"] is True
    assert d._weather_cache.ttl == 900
    assert d._weather_cache.stale_max_age == 1800


@pytest.mark.asyncio
async def test_reload_behavior_bad_config_is_reported_not_raised(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[behavior\nnot toml", encoding="utf-8")

    d = Daemon(Config(), config_path=cfg_file)
    before = d.cfg.behavior

    reply = await d._on_query({"command": "reload_behavior"})

    assert reply["ok"] is False
    assert d.cfg.behavior is before  # untouched on failure


def test_tone_cli_rewrites_only_the_humor_line(tmp_path, monkeypatch):
    """cmd_tone must flip the number in place and preserve every comment —
    users hand-edit this file."""
    from jarvis_cli import install as install_mod

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        "# my precious comments\n"
        "[behavior]\n"
        "# 0=deadpan, 1=light wit, 2=MCU Jarvis, 3=Tony-mode\n"
        "humor_level = 1\n"
        "events = [\n  \"permission_prompt\",\n]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(install_mod, "DEFAULT_CONFIG_PATH", cfg_file)
    # Daemon unreachable in tests: the reload reply is None and cmd_tone
    # reports "applies on next start" — still exit 0.
    monkeypatch.setattr(
        "jarvis_cli.hook_client._request_reply", lambda *a, **k: None,
    )

    rc = install_mod.cmd_tone(
        argparse.Namespace(level=3, quiet=True),
    )

    assert rc == 0
    text = cfg_file.read_text(encoding="utf-8")
    assert "humor_level = 3" in text
    assert "# my precious comments" in text
    assert "# 0=deadpan" in text
    assert len(re.findall(r"humor_level", text)) == 1


def test_tone_cli_errors_when_key_missing(tmp_path, monkeypatch):
    from jarvis_cli import install as install_mod

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text("[behavior]\n", encoding="utf-8")
    monkeypatch.setattr(install_mod, "DEFAULT_CONFIG_PATH", cfg_file)

    rc = install_mod.cmd_tone(argparse.Namespace(level=2, quiet=True))

    assert rc == 1
    assert "humor_level" not in cfg_file.read_text(encoding="utf-8")
