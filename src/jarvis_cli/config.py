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
    model_dir: str = "~/.jarvis-cli/models/xtts-v2"
    ref_audio_zh: str = "~/.jarvis-cli/voices/jarvis_zh.wav"
    ref_audio_en: str = "~/.jarvis-cli/voices/jarvis_en.wav"
    device: str = "mps"
    # XTTS GPT decoder sampling temperature. Library default 0.75 is too
    # high for our use case — short Jarvis-toned commands suffer audible
    # word repetition / pacing drift across takes. 0.5 cuts variance to
    # the point where successive synths of the same line are consistent.
    temperature: float = 0.5
    # XTTS's GPT learned an asymmetry from training data: short utterances
    # are delivered slowly (with pauses for emphasis), long utterances flow
    # quickly. A single `speed` multiplier therefore over-speeds long text
    # while still feeling sluggish on short status lines. We split the knob:
    # texts under `short_threshold_chars` get `speed_short` (faster),
    # longer ones get `speed_long` (closer to 1.0).
    speed_short: float = 1.30
    speed_long: float = 1.00
    short_threshold_chars: int = 60


@dataclass
class ElevenLabsConfig:
    api_key_env: str = "ELEVENLABS_API_KEY"
    voice_id: str = ""
    model: str = "eleven_turbo_v2_5"


@dataclass
class CosyVoiceConfig:
    """CosyVoice 3 (FunAudioLLM) via the Apache-2.0 Rust+Candle binding
    `cosyvoice3.rs`. Apple Silicon Metal acceleration; no PyTorch needed.

    Quality outperforms XTTS-v2 on speaker similarity in our A/B; the
    permissive license also clears the OSS path that XTTS's CPML blocks.
    """
    model_dir: str = "~/.jarvis-cli/models/cosyvoice3-0.5b-candle"
    ref_audio_zh: str = "~/.jarvis-cli/voices/jarvis_zh.wav"
    ref_audio_en: str = "~/.jarvis-cli/voices/jarvis_en.wav"
    # Transcript of each ref audio. When provided, the provider routes
    # through inference_zero_shot (which uses the transcript to ground the
    # LLM and prevent the double-take loop that cross_lingual mode falls
    # into on short utterances). Leave blank to use cross_lingual instead.
    ref_text_zh: str = ""
    ref_text_en: str = ""
    # CFM sampling steps for the flow decoder. Library default is 10;
    # we drop to 5 because the flow-decoder cost scales linearly with
    # steps and the quality delta on Bettany short lines is sub-perceptible
    # in A/B testing. Bump back to 10 if you ever hear artifacts.
    n_timesteps: int = 5
    # Double-take detection (see tts/duration_guard.py). After each synth we
    # compare its duration against the text's clean baseline (median of a
    # rolling window of recent clean takes); a synth running past baseline x
    # duration_ratio_threshold is a repeat and gets retried. `fallback_cps`
    # drives the chars/cps estimate used before a per-text baseline exists.
    # (An SSM self-similarity approach was tried and discarded — on real audio
    # it couldn't separate a true double-take from the ordinary self-similarity
    # of normal long speech.) `save_synth_samples` dumps each synth + metadata
    # to sample_dir for offline analysis.
    #
    # Threshold is 1.5 against the MEDIAN baseline: clean takes vary ±30-40%, so
    # a normal take lands within ~1.3x of the median while a full double-take is
    # ~2x — 1.5 sits cleanly between. (An earlier 1.35 against the per-text
    # MINIMUM mis-flagged ~30% of clean takes, because the min was the fastest
    # fluke, not the typical length.) Trade-off: a partial double-take near 1.4x
    # may pass — duration alone cannot separate it from clean jitter.
    duration_ratio_threshold: float = 1.5
    fallback_cps: float = 12.0
    max_synth_attempts: int = 4
    save_synth_samples: bool = False
    sample_dir: str = "~/.jarvis-cli/cache/samples"
    duration_baseline_path: str = "~/.jarvis-cli/cache/duration_baseline.json"


@dataclass
class PiperConfig:
    """Piper TTS (rhasspy) via the MIT `piper-tts` wheel: ONNX, CPU-only, no
    PyTorch. Unlike the CosyVoice zero-shot clone, Piper renders from a FIXED
    single-speaker model, so the speaker/accent is baked into the weights and
    cannot drift (the CosyVoice 0.5B clone intermittently drifted to an Indian
    accent) and there is no flow-decoder double-take to guard against. Warm
    per-utterance RTF is ~0.03 (≈30x realtime) since the model stays resident.

    Voices are `<name>.onnx` (+ `.onnx.json`) under `data_dir`; fetch with
    `python -m piper.download_voices <name> --data-dir <data_dir>`.
    """
    data_dir: str = "~/.jarvis-cli/models/piper"
    # British male butler voice — the default Jarvis identity is English.
    # Swap to a JARVIS-tuned voice (e.g. jgkawell/jarvis on HF) for closer timbre.
    voice_en: str = "en_GB-alan-medium"
    voice_zh: str = "zh_CN-huayan-medium"


@dataclass
class TTSConfig:
    provider: str = "xtts"
    fallback: str = "say"
    xtts: XTTSConfig = field(default_factory=XTTSConfig)
    elevenlabs: ElevenLabsConfig = field(default_factory=ElevenLabsConfig)
    cosyvoice: CosyVoiceConfig = field(default_factory=CosyVoiceConfig)
    piper: PiperConfig = field(default_factory=PiperConfig)


@dataclass
class PrivacyConfig:
    cloud_redaction: bool = True


@dataclass
class SessionBriefingConfig:
    """Iron-Man-style opening briefing on new CC/Codex sessions:
    greeting + local time + weather, English voice, no LLM round-trip.
    """
    enabled: bool = True
    # City queried against wttr.in. Empty = derive from timezone tail
    # (`Asia/Shanghai` → "Shanghai"). Override to pin a location when
    # the timezone is a continent root or you're abroad on a VPN.
    city: str = ""
    # How long a single weather lookup is reused across briefings — keeps
    # us off wttr.in if you open ten sessions in two minutes.
    weather_ttl_seconds: int = 600
    # Floor between briefings. 0 = every session_start speaks. Bump up if
    # you open many tabs and find the chorus tiresome.
    min_interval_seconds: int = 0
    # HTTP timeout for the weather fetch. Briefing falls back to a
    # time-only line if this elapses — we never block the worker on it.
    weather_timeout_seconds: float = 3.0


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
            "session_start",
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
    # How much wit Jarvis allows himself, 0-3. Plumbed into both the phrase
    # router system prompt and the session-start briefing prompt.
    #   0 — deadpan formal butler (no jokes)
    #   1 — hint of dry wit (default for first-time users)
    #   2 — MCU Jarvis: dry banter, witty asides
    #   3 — Tony-mode: openly sardonic, never sycophantic
    # Out-of-range values are clamped on load by `load_config`.
    humor_level: int = 1
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    session_briefing: SessionBriefingConfig = field(default_factory=SessionBriefingConfig)


@dataclass
class PathsConfig:
    socket: str = "~/.jarvis-cli/jarvis.sock"
    log: str = "~/.jarvis-cli/daemon.log"
    missed_log: str = "~/.jarvis-cli/missed.log"


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
    cfg.tts.cosyvoice.model_dir = expanduser(cfg.tts.cosyvoice.model_dir)
    cfg.tts.cosyvoice.ref_audio_zh = expanduser(cfg.tts.cosyvoice.ref_audio_zh)
    cfg.tts.cosyvoice.ref_audio_en = expanduser(cfg.tts.cosyvoice.ref_audio_en)
    # Clamp humor_level rather than rejecting — a typo in config shouldn't
    # leave the daemon refusing to start.
    cfg.behavior.humor_level = max(0, min(3, int(cfg.behavior.humor_level)))
    return cfg


DEFAULT_CONFIG_PATH = expanduser("~/.jarvis-cli/config.toml")
