"""Tests for the double-take sample summarizer (duration-ratio monitoring)."""
from __future__ import annotations

from jarvis_cli.tts.sample_stats import summarize


def test_summarize_groups_by_text_and_reports_ratio_spread():
    """Group samples by text; most-spoken texts surface first, and per-group
    we report the duration-ratio spread plus how many were flagged repeats."""
    records = [
        {"text": "hi", "ratio": 1.0, "is_repeat": False},
        {"text": "hi", "ratio": 2.0, "is_repeat": True},
        {"text": "bye", "ratio": 1.1, "is_repeat": False},
    ]

    s = summarize(records)

    assert s["total"] == 3
    groups = {g["text"]: g for g in s["groups"]}
    assert groups["hi"]["n"] == 2
    assert groups["hi"]["min"] == 1.0
    assert groups["hi"]["max"] == 2.0
    assert groups["hi"]["repeats"] == 1
    assert groups["bye"]["n"] == 1
    assert groups["bye"]["repeats"] == 0
    # Most-seen text first.
    assert s["groups"][0]["text"] == "hi"


def test_summarize_skips_records_without_ratio():
    """Pre-duration-era samples (no ratio field) are ignored, not crashed on."""
    records = [
        {"text": "hi", "ratio": 1.0, "is_repeat": False},
        {"text": "old", "ssm_score": 0.9},  # legacy sample, no ratio
    ]
    s = summarize(records)
    assert s["total"] == 1
    assert [g["text"] for g in s["groups"]] == ["hi"]


def test_summarize_handles_no_samples():
    s = summarize([])
    assert s["total"] == 0
    assert s["groups"] == []
