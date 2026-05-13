"""ElevenLabs cloud TTS provider."""
from __future__ import annotations

import os
from pathlib import Path

import httpx

from ...config import ElevenLabsConfig
from ...types import Lang
from .base import TTSProvider


class ElevenLabsProvider(TTSProvider):
    name = "elevenlabs"

    def __init__(self, cfg: ElevenLabsConfig) -> None:
        self.cfg = cfg

    async def synthesize(self, text: str, lang: Lang, out_path: Path) -> Path:
        key = os.getenv(self.cfg.api_key_env)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        if not self.cfg.voice_id:
            raise RuntimeError("ElevenLabs voice_id is not configured")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(
            base_url="https://api.elevenlabs.io", timeout=15.0
        ) as client:
            r = await client.post(
                f"/v1/text-to-speech/{self.cfg.voice_id}",
                headers={
                    "xi-api-key": key,
                    "accept": "audio/mpeg",
                    "content-type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.cfg.model,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
            )
            r.raise_for_status()
            out_path.write_bytes(r.content)
        return out_path

    async def healthcheck(self) -> bool:
        return bool(os.getenv(self.cfg.api_key_env)) and bool(self.cfg.voice_id)
