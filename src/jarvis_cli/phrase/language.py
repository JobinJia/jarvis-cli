"""Project-language detection: looks at CLAUDE.md, README.md in cwd."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from loguru import logger

from ..types import Lang

DetectorFactory.seed = 0  # deterministic

_SOURCES = ["CLAUDE.md", "AGENTS.md", "README.md", "README"]
_DEFAULT: Lang = "en"
_SAMPLE_CHARS = 500


def _read_sample(cwd: Path) -> str | None:
    for name in _SOURCES:
        p = cwd / name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="ignore")[:_SAMPLE_CHARS]
            except OSError:
                continue
    return None


@lru_cache(maxsize=64)
def detect_for(cwd: str | None) -> Lang:
    """Return 'zh' or 'en' for the project at `cwd`.

    Default 'en' when no signal can be extracted; switches to 'zh' only when
    CLAUDE.md / AGENTS.md / README first 500 chars look Chinese.
    """
    if not cwd:
        return _DEFAULT
    try:
        sample = _read_sample(Path(cwd))
    except OSError:
        return _DEFAULT
    if not sample or not sample.strip():
        return _DEFAULT
    try:
        code = detect(sample)
    except LangDetectException:
        return _DEFAULT
    if code.startswith("zh"):
        return "zh"
    if code.startswith("en"):
        return "en"
    logger.debug("Unmapped langdetect result {!r}, falling back to {}", code, _DEFAULT)
    return _DEFAULT
