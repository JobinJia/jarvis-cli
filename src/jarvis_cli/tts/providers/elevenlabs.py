"""ElevenLabs cloud TTS provider."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from ...config import ElevenLabsConfig, resolve_api_key
from ...types import Lang
from .base import TTSProvider


def _quota_exhausted_message(body: bytes) -> str | None:
    """If `body` is an ElevenLabs quota_exceeded JSON error, return a single-
    line human message; otherwise None. ElevenLabs wraps quota errors in
    HTTP 401, which is indistinguishable from auth failures at the status
    layer — only the body distinguishes them.
    """
    try:
        detail = json.loads(body).get("detail")
    except (ValueError, AttributeError):
        return None
    if not isinstance(detail, dict):
        return None
    if detail.get("code") != "quota_exceeded" and detail.get("status") != "quota_exceeded":
        return None
    msg = detail.get("message") or "quota exceeded"
    return f"ElevenLabs quota exhausted: {msg}"


class ElevenLabsProvider(TTSProvider):
    name = "elevenlabs"
    supports_streaming = True

    def __init__(self, cfg: ElevenLabsConfig) -> None:
        self.cfg = cfg

    def _resolve(self, voice_id: str | None) -> tuple[str, str]:
        key = resolve_api_key(self.cfg)
        if not key:
            raise RuntimeError(f"{self.cfg.api_key_env} not set")
        effective_voice = voice_id or self.cfg.voice_id
        if not effective_voice:
            raise RuntimeError("ElevenLabs voice_id is not configured")
        return key, effective_voice

    def _body(self, text: str) -> dict:
        return {
            "text": text,
            "model_id": self.cfg.model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        out_path: Path,
        voice_id: str | None = None,
    ) -> Path:
        key, effective_voice = self._resolve(voice_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(
            base_url="https://api.elevenlabs.io", timeout=15.0
        ) as client:
            r = await client.post(
                f"/v1/text-to-speech/{effective_voice}",
                headers={
                    "xi-api-key": key,
                    "accept": "audio/mpeg",
                    "content-type": "application/json",
                },
                json=self._body(text),
            )
            if r.status_code == 401:
                quota = _quota_exhausted_message(r.content)
                if quota is not None:
                    raise RuntimeError(quota)
            r.raise_for_status()
            out_path.write_bytes(r.content)
        return out_path

    async def stream(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        key, effective_voice = self._resolve(voice_id)
        async with httpx.AsyncClient(
            base_url="https://api.elevenlabs.io", timeout=30.0
        ) as client:
            async with client.stream(
                "POST",
                f"/v1/text-to-speech/{effective_voice}/stream",
                headers={
                    "xi-api-key": key,
                    "accept": "audio/mpeg",
                    "content-type": "application/json",
                },
                json=self._body(text),
            ) as response:
                if response.status_code == 401:
                    body = await response.aread()
                    quota = _quota_exhausted_message(body)
                    if quota is not None:
                        raise RuntimeError(quota)
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    if chunk:
                        yield chunk

    async def healthcheck(self) -> bool:
        return bool(resolve_api_key(self.cfg)) and bool(self.cfg.voice_id)
