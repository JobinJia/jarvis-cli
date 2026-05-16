"""TOML-backed config with safe defaults. Layered: file > defaults."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DeepSeekConfig:
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 5.0


@dataclass
class AnthropicConfig:
    api_key_env: str = "ANTHROPIC_API_KEY"
    model: str = "claude-haiku-4-5-20251001"
    timeout_seconds: float = 5.0


@dataclass
class OpenAIConfig:
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 5.0


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout_seconds: float = 10.0


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    fallback: str = "ollama"
    deepseek: DeepSeekConfig = field(default_factory=DeepSeekConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class XTTSConfig:
    model_dir: str = "~/.jarvis-cc/models/xtts-v2"
    ref_audio_zh: str = "~/.jarvis-cc/voices/jarvis_zh.wav"
    ref_audio_en: str = "~/.jarvis-cc/voices/jarvis_en.wav"
    device: str = "mps"


@dataclass
class ElevenLabsConfig:
    api_key_env: str = "ELEVENLABS_API_KEY"
    voice_id: str = ""
    model: str = "eleven_turbo_v2_5"


@dataclass
class TTSConfig:
    provider: str = "xtts"
    fallback: str = "say"
    xtts: XTTSConfig = field(default_factory=XTTSConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)


@dataclass
class PrivacyConfig:
    cloud_redaction: bool = True


@dataclass
class BehaviorConfig:
    dedup_window_seconds: int = 10
    queue_max_size: int = 5
    # User-facing language switch for what Jarvis SPEAKS. Accepted values:
    # "en" (English, default — matches the user's chosen British voice
    # identity), "zh" (Chinese), or "auto" (pick per-event from content).
    # Honored by both the LLM phrase path and the hook AskUserQuestion path.
    voice_language: str = "en"
    events: list[str] = field(
        default_factory=lambda: [
            "permission_prompt",
            "idle_prompt",
            "elicitation_dialog",
            "ask_user_question",
        ]
    )
    # DEPRECATED: kept so old config.toml files don't error on load. Not read
    # at runtime; replaced by phrase_target_chars + phrase_hard_cap below.
    phrase_max_chars: int = 30
    phrase_target_chars: int = 70
    phrase_hard_cap: int = 120
    # When True (default), the hook sends a cancel signal on UserPromptSubmit /
    # PostToolUse so the daemon stops any in-flight audio for that session.
    cancel_on_user_action: bool = True
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


@dataclass
class PathsConfig:
    socket: str = "~/.jarvis-cc/jarvis.sock"
    log: str = "~/.jarvis-cc/daemon.log"
    missed_log: str = "~/.jarvis-cc/missed.log"


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def expanduser(p: str) -> str:
    return str(Path(os.path.expanduser(p)))


def _merge(dst, src: dict) -> None:
    """Shallow-recursive merge: dict src into dataclass dst, mutating dst."""
    for key, val in src.items():
        if not hasattr(dst, key):
            continue
        cur = getattr(dst, key)
        if isinstance(val, dict) and hasattr(cur, "__dataclass_fields__"):
            _merge(cur, val)
        else:
            setattr(dst, key, val)


def load_config(path: str | Path) -> Config:
    """Load TOML config from `path`, fall back to defaults if missing/empty.

    Path-like values under `[paths]` and `[tts.xtts]` get `~` expanded.
    """
    cfg = Config()
    p = Path(path)
    if p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        _merge(cfg, data)

    cfg.paths.socket = expanduser(cfg.paths.socket)
    cfg.paths.log = expanduser(cfg.paths.log)
    cfg.paths.missed_log = expanduser(cfg.paths.missed_log)
    cfg.tts.xtts.model_dir = expanduser(cfg.tts.xtts.model_dir)
    cfg.tts.xtts.ref_audio_zh = expanduser(cfg.tts.xtts.ref_audio_zh)
    cfg.tts.xtts.ref_audio_en = expanduser(cfg.tts.xtts.ref_audio_en)
    return cfg


DEFAULT_CONFIG_PATH = expanduser("~/.jarvis-cc/config.toml")
