from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis_cc.tts.providers.say import SayProvider


@pytest.mark.asyncio
async def test_say_writes_wav_to_path(tmp_path: Path):
    p = SayProvider()
    # We don't actually want to spawn `say` in CI; mock subprocess.
    out_path = tmp_path / "out.aiff"
    with patch("jarvis_cc.tts.providers.say.asyncio.create_subprocess_exec") as mock_exec:
        async def _fake(*args, **kwargs):
            # Pretend `say` succeeded and wrote a file
            (out_path).write_bytes(b"FAKE-AUDIO")

            class _P:
                returncode = 0

                async def communicate(self):
                    return (b"", b"")

                async def wait(self):
                    return 0

            return _P()

        mock_exec.side_effect = _fake

        audio = await p.synthesize("hello sir", lang="en", out_path=out_path)
    assert audio == out_path
    assert audio.read_bytes() == b"FAKE-AUDIO"
