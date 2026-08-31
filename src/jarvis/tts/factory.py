"""Provider construction, shared by the daemon and the synthesis worker.

This lives here rather than in `daemon.main` so the worker child can build the
same provider from the same config without importing the daemon — it has no
business pulling in the unix-socket server, the notify stack or the retrieval
index just to synthesize a sentence.
"""
from __future__ import annotations

from loguru import logger

from ..config import Config
from .providers.base import TTSProvider
from .providers.cosyvoice import CosyVoiceProvider
from .providers.elevenlabs import ElevenLabsProvider
from .providers.piper import PiperProvider
from .providers.say import SayProvider
from .providers.xtts import XTTSProvider

#: Providers worth isolating in a recyclable child process. Both hold a
#: multi-GB model AND leak native memory per synthesis — measured 2026-08-26,
#: XTTS grows MALLOC_NANO + MALLOC_SMALL by ~40 MB on every utterance, linearly
#: and with no plateau across 18 runs, while the Python object count stays
#: pinned at 758k. That combination places the leak inside torch/coqui rather
#: than our code, so no amount of releasing on our side returns it: only ending
#: the process does. Everything else here is cheap enough to stay in-process.
HEAVY_PROVIDERS = frozenset({"xtts", "cosyvoice"})

#: name -> (class, attribute on TTSConfig holding that provider's settings).
#: The class is exposed separately from construction so the daemon can read a
#: provider's streaming contract (supports_streaming / stream_pcm /
#: stream_input_args) for a provider that actually runs in a child process.
_PROVIDERS: dict[str, tuple[type[TTSProvider], str | None]] = {
    "xtts": (XTTSProvider, "xtts"),
    "cosyvoice": (CosyVoiceProvider, "cosyvoice"),
    "piper": (PiperProvider, "piper"),
    "elevenlabs": (ElevenLabsProvider, "elevenlabs"),
    "say": (SayProvider, None),
}


def provider_class(name: str) -> type[TTSProvider] | None:
    entry = _PROVIDERS.get(name)
    return entry[0] if entry else None


def make_provider(name: str, cfg: Config) -> TTSProvider | None:
    """Build the named provider, or None when the name is unknown/empty."""
    entry = _PROVIDERS.get(name)
    if entry is None:
        if name:  # empty string means "no fallback configured", not a typo
            logger.warning("Unknown TTS provider {!r}; skipping in chain", name)
        return None
    cls, cfg_attr = entry
    if cfg_attr is None:
        return cls()
    return cls(getattr(cfg.tts, cfg_attr))
