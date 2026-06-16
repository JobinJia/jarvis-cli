# CosyVoice double-take detection & fixed-phrase cache — design

Date: 2026-06-07
Status: approved (brainstorming) — pending implementation plan

## Background

CosyVoice 3 (`cosyvoice3.rs`) occasionally synthesizes short utterances as a
"double-take": the model's flow decoder emits the line, or its tail, twice.
This is a probabilistic artifact of the decoder, not config-fixable, and
independent of `n_timesteps`.

The current mitigation in `CosyVoiceProvider.synthesize` measures speech rate
(`cps = chars / audio_duration`) after synthesis and retries up to 3 times when
`cps < 9.0`, but only for inputs `<= 80` chars (`_VALIDATION_MAX_CHARS`).

## Problem (evidence from `~/.jarvis-cli/daemon.log`)

1. **`cps` cannot separate "repeat" from "slow speech".** The 33-char phrase
   `Sir, Claude awaits your guidance.` has a clean main peak at cps 13–16, a
   sharp full-double-take spike at `duration=5.60s / cps=5.9` (13 occurrences),
   AND a large band at cps 10–12. The 10–12 band is *partial* double-take
   (only the tail repeats) — it stretches duration just enough to dodge the
   `cps < 9` trigger. These are the leaks the user still hears.

2. **The 80-char cap leaves long dynamic lines completely unguarded.** Logged
   example: `chars=147 cps=8.0 duration=18.44s` ("Sir, he asks how to
   proceed — option one…") — clearly abnormal, but skipped because
   `147 > 80`. LLM-generated prompt lines and PR summaries fall here.

3. The retry logic only became live after the daemon restart at
   2026-06-06 13:04; the 13 full double-takes above all predate it. Post-restart
   the full double-takes are caught, but partial ones (#1) and long lines (#2)
   still leak.

## Root cause

Two layers:
- **Model layer**: probabilistic double-take in the flow decoder (unfixable).
- **Detection layer**: `cps` is an indirect proxy. It conflates repetition with
  slow speech, and its 80-char cap disables it for long lines.

## Goals / Non-goals

Goals:
- Detect double-takes (full and partial, short and long) by their *structure*,
  not by an indirect speed proxy.
- Eliminate double-takes entirely for high-frequency, fully-fixed phrases.
- Tune the detector on real, labeled samples rather than guessed thresholds.

Non-goals:
- Changing the TTS model or its inference params (dead end — verified).
- ASR round-trip verification (kept as a future upgrade path only).
- Caching dynamic or templated text (LLM-generated lines, briefings).

## Approach (chosen: option A)

Fixed-phrase audio cache + self-similarity-matrix (SSM) repeat detection,
replacing the `cps` heuristic. No new dependency (numpy is already pulled in by
the `cosyvoice` extra).

## Component design

Three single-purpose, independently testable units.

### 1. `tts/doubletake.py` (new, pure functions)

`detect_repeat(audio: np.ndarray, sample_rate: int) -> RepeatScore`

- Raw-waveform autocorrelation is not robust to speech (pitch/phase). Instead
  work on a feature-frame sequence:
  1. **Feature extraction** (hand-written STFT, numpy only): frame the signal
     (~25 ms window, ~10 ms hop), apply a window, `np.fft.rfft`, take the log
     magnitude spectrum → a `[frames × bins]` sequence. No mel filterbank
     needed; log-magnitude frames already capture phoneme/timbre repetition.
  2. **Self-similarity matrix** `S[i,j] = cosine(frame_i, frame_j)`. A few
     hundred frames → O(n²) is a few milliseconds.
  3. **Repeat search**: a repeated segment forms a bright diagonal *off* the
     main diagonal. For each offset `lag >= min_repeat_frames` (~0.25 s),
     compute the mean similarity along that diagonal; if some lag's diagonal
     mean exceeds the threshold over a sufficient contiguous length → repeat.
- Returns: score, boolean verdict, and the repeat segment's offset/length
  (distinguishes full vs tail-only repeat; useful for debugging).
- Pure, no provider dependency. Unit-testable by feeding synthetic/concatenated
  audio arrays.

Why this catches what `cps` misses: cps reads average speed; a tail repeat
barely moves it (lands in 10–12). The SSM diagonal reads "did a segment occur
again" — full repeats (diagonal spanning the whole clip) and partial tail
repeats (a shorter diagonal) both show up, independent of speed and length.

### 2. `tts/cache.py` (new)

`get_or_synth(text, lang, synth_fn) -> Path`

- Scope: only **fully-fixed phrases with no placeholders** (e.g. the ~3 literal
  templates in `phrase/templates.py` such as `Sir, Claude awaits your
  guidance.`). Templated (`{tool}`) and briefing "fixed-opener + dynamic body"
  lines are NOT cached — splicing cached audio onto live audio breaks
  CosyVoice's prosody continuity.
- Generation: lazy. On first encounter of a cacheable text, synthesize under a
  *stricter* standard (synthesize a few times, keep the one with the lowest SSM
  score) before writing to disk, so the cache only ever holds clean samples.
- Key: `hash(text + lang + reference-audio fingerprint)` — changing the voice
  reference invalidates automatically.
- Location: `~/.jarvis-cli/cache/tts/`. Hit → play directly, zero latency.

### 3. `tts/providers/cosyvoice.py` (modified)

- `_synth_once` returns the audio array (currently returns only duration), so
  `detect_repeat` reuses the same data without re-reading the file.
- `synthesize`'s retry loop replaces the `cps < 9` predicate with
  `detect_repeat(...)`. `cps` is demoted to a logged-only auxiliary signal.
- **Remove `_VALIDATION_MAX_CHARS`**: SSM works on long lines too, so all
  syntheses are checked (fixes problem #2). Per-synth SSM cost is negligible.
- Retry cap configurable (raise to ~4 given "accuracy first, can wait").

### 4. Sampling & threshold calibration (explicit step, not a TODO)

No persisted samples exist today, so calibration is part of the plan:

1. Add `tts.save_synth_samples` config flag (default off). When on, each synth
   writes the wav + metadata (text / duration / cps / SSM score) to
   `cache/samples/`.
2. Run normally for a few days — naturally accumulates positives and negatives
   (double-take is ~12%).
3. Semi-automatic labeling: full repeats pre-labeled via "duration ≈ 2× baseline";
   remaining borderline cases labeled by listening (repeat / clean).
4. Plot the SSM-score distribution; pick the threshold under a "tolerable
   false-positive (extra-retry) rate" constraint — biased toward **recall**
   (see asymmetry below). Write it into config defaults, turn sampling off.

**Cost asymmetry drives the threshold direction.** A miss → repeat is *played*
(user-visible, bad). A false alarm → one *extra synthesis* (a few seconds wait).
With "accuracy first, can wait", the threshold is tuned for high recall,
tolerating extra retries rather than letting repeats through.

**Bootstrap before calibration**: ship with a conservative threshold (catches
at least the strong-diagonal full repeats current `cps` already caught, plus
clear partial repeats) while sampling runs. Tighten to the recall-biased value
after calibration. No regression on day one.

### 5. Config additions

- `tts.save_synth_samples: bool = false`
- `tts.doubletake_threshold: float` (conservative bootstrap default)
- `tts.max_synth_attempts: int = 4`

## Data flow

```
event → daemon → engine.synthesize
   → cache hit? ──yes→ play
   └─no→ cosyvoice.synthesize
          loop { _synth_once → detect_repeat → clean? play : retry (cap N) }
          → if fixed phrase: store in cache → play
```

## Error handling (always degrade toward "make sound")

- SSM fails (clip too short / all-silence) → treat as pass, log, don't block.
- Cache read/write failure → fall back to live synthesis.
- Retry cap exhausted still flagged → play the last take (matches current
  behavior, guarantees output) + warning.

## Testing strategy

- `doubletake`: pure function. Feed clean audio (no significant off-diagonal)
  and concatenated/repeated audio (strong diagonal); assert verdicts. Include a
  tail-only-repeat fixture.
- `cache`: hit / miss / invalidation (changed reference fingerprint).
- `provider`: mock `_synth_once` to return preset audio; assert retry triggers
  and cache writes on fixed phrases.

## Implementation order (dependency-aware, ships incrementally)

1. `doubletake.py` (SSM) + unit tests.
2. Wire into `cosyvoice.synthesize` (conservative threshold) + sampling flag.
   ← shippable here; only adds coverage, never removes.
3. `cache.py` fixed-phrase cache.
4. Sample for a few days → calibrate threshold → freeze defaults, disable
   sampling.
5. (Optional, later) briefing fixed-opener splice cache; ASR upgrade path.

## Risks & trade-offs

- **Threshold quality depends on the sample set.** Mitigated by the explicit
  calibration step and conservative bootstrap; recall-biased per cost asymmetry.
- **Hand-written STFT in numpy** rather than librosa/scipy — keeps zero new
  deps; log-magnitude frames are sufficient for repeat structure.
- **Cache scope is intentionally narrow** (fully-fixed phrases only). Briefings
  and templated lines rely on SSM detection, accepted to avoid prosody seams.
- **ASR (option B) deferred**: most accurate but a heavy resident dependency;
  revisit only if SSM recall proves insufficient after calibration.
