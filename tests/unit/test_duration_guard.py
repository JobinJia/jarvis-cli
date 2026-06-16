"""Tests for the per-text duration baseline double-take guard.

A double-take makes a synth run much longer than the same text's clean
duration. The baseline is the MEDIAN of a rolling window of recent clean
durations per exact text — robust to a single fast/slow outlier, where the
old historical-minimum baseline let one fast fluke permanently flag every
normal take. A synth exceeding baseline x ratio is a repeat. Unseen text
falls back to a chars/cps estimate.
"""
from __future__ import annotations

from pathlib import Path

from jarvis_cli.tts.duration_guard import DurationBaseline


def _guard(tmp_path: Path) -> DurationBaseline:
    return DurationBaseline(tmp_path / "baseline.json")


def test_baseline_uses_median_not_min(tmp_path: Path) -> None:
    """A single fast outlier must NOT drag the baseline down. The old min-based
    baseline did, so every later normal take looked like a double-take."""
    g = _guard(tmp_path)
    for d in (2.4, 2.6, 2.5, 1.5, 2.5):   # 1.5 is a fast fluke
        g.record("hello", d)
    v = g.check("hello", 2.5, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.expected == 2.5            # median, not the 1.5 minimum
    assert not v.is_repeat              # a normal 2.5s take stays clean


def test_baseline_window_is_bounded_and_tracks_recent(tmp_path: Path) -> None:
    """The window keeps only recent takes, so the baseline follows the model's
    current behavior instead of being pinned by ancient samples."""
    g = _guard(tmp_path)
    for _ in range(20):
        g.record("hello", 1.0)
    for _ in range(20):
        g.record("hello", 3.0)          # evicts the old 1.0s takes
    v = g.check("hello", 3.0, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.expected == 3.0


def test_loads_legacy_float_format(tmp_path: Path) -> None:
    """Old baselines stored one float per text; load them as a 1-sample
    window so the existing cache keeps working across the upgrade."""
    path = tmp_path / "baseline.json"
    path.write_text('{"hello": 2.0}')
    g = DurationBaseline(path)
    v = g.check("hello", 2.0, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.expected == 2.0


def test_record_rejects_implausibly_slow_take_when_gated(tmp_path: Path) -> None:
    """The escape valve records a forced take only if its pace is plausibly
    clean. A take slower than min_cps is itself a double-take and must not
    teach the baseline."""
    g = _guard(tmp_path)
    text = "x" * 30
    g.record(text, 6.0, min_cps=9.0)    # 30/6 = 5 cps → too slow, rejected
    g.record(text, 2.5, min_cps=9.0)    # 30/2.5 = 12 cps → plausible, kept
    v = g.check(text, 2.5, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.expected == 2.5


def test_check_flags_duration_over_baseline(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    g.record("hello", 2.0)
    v = g.check("hello", 3.0, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.is_repeat        # 3.0 / 2.0 = 1.5 > 1.35
    assert v.ratio == 1.5


def test_check_accepts_normal_duration(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    g.record("hello", 2.0)
    v = g.check("hello", 2.2, ratio_threshold=1.35, fallback_cps=12.0)
    assert not v.is_repeat     # 1.1 < 1.35


def test_unseen_text_uses_char_fallback(tmp_path: Path) -> None:
    """No baseline yet → expected = chars / fallback_cps. A clearly doubled
    take (cps far below ~9) is flagged; a normal one is not."""
    g = _guard(tmp_path)
    text = "x" * 24          # expected = 24/12 = 2.0s
    doubled = g.check(text, 4.0, ratio_threshold=1.35, fallback_cps=12.0)
    normal = g.check(text, 2.0, ratio_threshold=1.35, fallback_cps=12.0)
    assert doubled.is_repeat   # 4.0 > 2.0*1.35
    assert not normal.is_repeat


def test_baseline_persists_across_instances(tmp_path: Path) -> None:
    g1 = _guard(tmp_path)
    g1.record("hello", 2.5)
    g2 = DurationBaseline(tmp_path / "baseline.json")   # reload from disk
    v = g2.check("hello", 2.5, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.expected == 2.5


def test_record_rejects_implausibly_short_take(tmp_path: Path) -> None:
    """A truncated/garbage take (impossibly fast cps — e.g. cancelled mid-synth)
    must not poison the baseline. Otherwise its tiny duration becomes the
    baseline and every later normal take looks like a double-take."""
    g = _guard(tmp_path)
    text = "Sir, an action is pending — your direction, please."  # 51 chars
    g.record(text, 0.2)   # 0.2s for 51 chars = 255 cps — impossible
    v = g.check(text, 4.0, ratio_threshold=1.35, fallback_cps=12.0)
    # 0.2s was rejected → falls back to char estimate (~4.25s), 4.0s is clean.
    assert v.expected != 0.2
    assert not v.is_repeat


def test_missing_baseline_file_is_empty(tmp_path: Path) -> None:
    g = DurationBaseline(tmp_path / "does_not_exist.json")
    # Falls back to char estimate, doesn't crash.
    v = g.check("hello", 1.0, ratio_threshold=1.35, fallback_cps=12.0)
    assert v.expected > 0
