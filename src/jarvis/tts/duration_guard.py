"""Per-text duration baseline for double-take detection.

CosyVoice occasionally "double-takes" — emitting a line, or its tail, twice in
one clip, which makes the audio run noticeably longer than the same text's
clean rendering. An earlier SSM (self-similarity) approach was tried and
discarded: on real samples it could not separate a true double-take from the
ordinary self-similarity of normal long speech (same cloned voice, repeated
function words, prosody) — clean and repeat scores overlapped completely.

What DID separate cleanly on the same samples was duration: a full double-take
runs ~2x the clean length, a partial one ~1.4x+. So we track the clean
duration per exact text and flag a synth that runs well past it.

- Baseline = MEDIAN of a rolling window of recent clean durations for a text.
  An earlier version used the historical MINIMUM, reasoning that double-takes
  only lengthen audio so the min is the cleanest estimate. That backfired:
  CosyVoice's clean takes vary ±30-40% in length, so the min captures the
  single fastest fluke, and with the ratio threshold sitting just above the
  typical take, ~30% of perfectly clean takes tripped it (observed on real
  samples). Worse, the min only ever ratcheted DOWN — once a text's baseline
  was set too low, every take read as a repeat, `record` (clean-only) never
  fired, and the baseline could never recover. The median over a rolling
  window is robust to a lone fast/slow outlier and tracks current behavior.
- Unseen text has no baseline → fall back to a chars/cps estimate. This is
  essentially the old "cps below ~9 is a double-take" heuristic, kept only as
  a coarse first-pass until a per-text baseline exists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from loguru import logger

# Faster than any real speech — a take above this cps is truncated/garbage
# (e.g. cancelled mid-synth), not a clean rendering. We refuse to learn a
# baseline from it; otherwise its tiny duration would make every later normal
# take look like a double-take.
_MAX_PLAUSIBLE_CPS = 22.0

# Below this chars-per-second a take is itself too slow to be a clean
# rendering — it IS a double-take (saying the line twice ~halves the cps from
# the ~10-15 cps clean band down to ~5-7). The retry loop's escape valve uses
# it to gate what it's willing to learn a baseline from. Matches the coarse
# "cps below ~9 is a double-take" heuristic the char fallback descends from.
_MIN_CLEAN_CPS = 9.0

# Rolling window size per text. Long enough for a stable median, short enough
# that the baseline follows the model's current behavior rather than ancient
# takes. Odd so the median is an actual observed value.
_WINDOW = 15


@dataclass(frozen=True)
class DurationVerdict:
    """Outcome of a baseline check. `expected` is the clean duration we
    compared against (recorded baseline or char estimate); `ratio` is
    duration / expected."""

    is_repeat: bool
    expected: float
    ratio: float


class DurationBaseline:
    """Per-text clean-duration baseline, persisted to a JSON file. Each text
    maps to a rolling window of recent clean durations; the baseline is their
    median."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data: dict[str, list[float]] = self._load()

    def _load(self) -> dict[str, list[float]]:
        """Load the per-text windows. Tolerates the legacy format that stored a
        single float per text (migrated to a 1-sample window)."""
        try:
            raw = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError, ValueError, AttributeError):
            return {}
        data: dict[str, list[float]] = {}
        for k, v in raw.items():
            if isinstance(v, list):
                samples = [float(x) for x in v]
            else:
                samples = [float(v)]  # legacy single-float baseline
            if samples:
                data[str(k)] = samples[-_WINDOW:]
        return data

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2)
            )
        except OSError as exc:
            logger.warning("duration baseline save failed: {}", exc)

    def check(
        self,
        text: str,
        duration: float,
        *,
        ratio_threshold: float,
        fallback_cps: float,
    ) -> DurationVerdict:
        """Judge whether `duration` indicates a double-take for `text`.

        Uses the median of the recorded per-text window if present, otherwise a
        chars/fallback_cps estimate. Read-only — call `record` to update the
        baseline after accepting a take.
        """
        window = self._data.get(text)
        expected = median(window) if window else 0.0
        if expected <= 0:
            expected = (len(text) / fallback_cps) if text else duration
        ratio = duration / expected if expected > 0 else 1.0
        return DurationVerdict(
            is_repeat=ratio > ratio_threshold, expected=expected, ratio=ratio,
        )

    def record(self, text: str, duration: float, *, min_cps: float = 0.0) -> None:
        """Append a take's duration to the text's rolling window.

        Call with no `min_cps` for confirmed-clean takes (those that already
        passed the duration check). The retry loop's escape valve passes
        `min_cps=_MIN_CLEAN_CPS` for a forced/uncertain take, so a take too slow
        to be clean (i.e. a real double-take) is not learned as a baseline.
        """
        if not text or duration <= 0:
            return
        cps = len(text) / duration
        if cps > _MAX_PLAUSIBLE_CPS:
            return  # implausibly fast → truncated/garbage take, ignore
        if cps < min_cps:
            return  # too slow → itself a double-take, don't learn from it
        window = self._data.setdefault(text, [])
        window.append(duration)
        del window[:-_WINDOW]  # keep only the most recent _WINDOW samples
        self._save()
