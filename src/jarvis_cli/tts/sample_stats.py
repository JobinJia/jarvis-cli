"""Summarize double-take sample metadata for monitoring.

When `tts.cosyvoice.save_synth_samples` is on, the provider drops a `.wav` plus
a `.json` sidecar (text, duration, expected, ratio, verdict) per synth into
`sample_dir`. This aggregates the sidecars by text and reports the duration
ratio (actual / expected-clean) spread, so you can see which texts run long
(high ratio = frequent double-takes) and sanity-check the baseline.

Run it with:

    python -m jarvis_cli.tts.sample_stats [sample_dir]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_records(sample_dir: Path) -> list[dict[str, Any]]:
    """Read every `*.json` sidecar in `sample_dir`, skipping unreadable ones."""
    records: list[dict[str, Any]] = []
    for jf in sorted(sample_dir.glob("*.json")):
        try:
            records.append(json.loads(jf.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Group records by text and report the duration-ratio spread per group.

    Groups are sorted by sample count descending, so the most-spoken texts
    surface first. Records without a `ratio` field (pre-duration-era samples)
    are skipped.
    """
    by_text: dict[str, list[float]] = defaultdict(list)
    repeats: dict[str, int] = defaultdict(int)
    total = 0
    for r in records:
        if "ratio" not in r:
            continue
        total += 1
        by_text[r["text"]].append(float(r["ratio"]))
        if r.get("is_repeat"):
            repeats[r["text"]] += 1

    groups: list[dict[str, Any]] = []
    for text, ratios in by_text.items():
        ordered = sorted(ratios)
        groups.append(
            {
                "text": text,
                "n": len(ordered),
                "repeats": repeats[text],
                "min": ordered[0],
                "median": ordered[len(ordered) // 2],
                "max": ordered[-1],
            }
        )
    groups.sort(key=lambda g: g["n"], reverse=True)
    return {"total": total, "groups": groups}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sample_dir",
        nargs="?",
        default="~/.jarvis-cli/cache/samples",
        help="directory of *.json sample sidecars",
    )
    args = ap.parse_args()
    sample_dir = Path(args.sample_dir).expanduser()
    if not sample_dir.is_dir():
        print(f"no sample dir yet: {sample_dir}")
        return

    summary = summarize(load_records(sample_dir))
    print(f"samples: {summary['total']}  distinct texts: {len(summary['groups'])}\n")
    print(f"{'n':>4} {'rep':>4}  {'min':>5} {'med':>5} {'max':>5}  text")
    for g in summary["groups"]:
        print(
            f"{g['n']:>4} {g['repeats']:>4}  {g['min']:>5.2f} {g['median']:>5.2f} "
            f"{g['max']:>5.2f}  {g['text'][:48]!r}"
        )


if __name__ == "__main__":
    main()
