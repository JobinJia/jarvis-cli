from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.tts.providers.say import SayProvider


def _fake_say_factory(out_path: Path, payload: bytes = b"FAKE-AUDIO"):
    """Build a fake create_subprocess_exec that records argv and pretends say wrote a file."""
    captured: dict = {"argv": None}

    async def _fake(*args, **kwargs):
        captured["argv"] = args
        out_path.write_bytes(payload)

        class _P:
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            async def wait(self):
                return 0

        return _P()

    return _fake, captured


@pytest.mark.asyncio
async def test_say_writes_to_aiff_path(tmp_path: Path):
    p = SayProvider()
    out_path = tmp_path / "out.aiff"
    fake, _ = _fake_say_factory(out_path)
    with patch("jarvis.tts.providers.say.asyncio.create_subprocess_exec", side_effect=fake):
        audio = await p.synthesize("hello sir", lang="en", out_path=out_path)
    assert audio == out_path
    assert audio.read_bytes() == b"FAKE-AUDIO"


@pytest.mark.asyncio
async def test_say_passes_data_format_for_wav_path(tmp_path: Path):
    """`say` won't write a real .wav unless --data-format is passed; otherwise
    it errors with 'Opening output file failed: fmt?'. Caller hands us a
    .wav tempfile (daemon main._worker), so we must request a WAV data format.
    """
    p = SayProvider()
    out_path = tmp_path / "out.wav"
    fake, captured = _fake_say_factory(out_path)
    with patch("jarvis.tts.providers.say.asyncio.create_subprocess_exec", side_effect=fake):
        await p.synthesize("hello sir", lang="en", out_path=out_path)
    argv = captured["argv"]
    assert any(
        isinstance(a, str) and a.startswith("--data-format=") for a in argv
    ), f"expected --data-format=... in argv for .wav output, got {argv!r}"


@pytest.mark.asyncio
async def test_say_omits_data_format_for_aiff_path(tmp_path: Path):
    """AIFF is say's native container; don't force a data format there."""
    p = SayProvider()
    out_path = tmp_path / "out.aiff"
    fake, captured = _fake_say_factory(out_path)
    with patch("jarvis.tts.providers.say.asyncio.create_subprocess_exec", side_effect=fake):
        await p.synthesize("hello sir", lang="en", out_path=out_path)
    argv = captured["argv"]
    assert not any(
        isinstance(a, str) and a.startswith("--data-format=") for a in argv
    ), f"unexpected --data-format for .aiff output: {argv!r}"
