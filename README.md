# jarvis

**English** | [简体中文](README.zh-CN.md)

A Jarvis-voiced notification layer for [Claude Code](https://claude.com/claude-code) and [Codex CLI](https://github.com/openai/codex).

When Claude Code or Codex CLI needs your attention — permission prompts, idle waits, MCP elicitation dialogs, AskUserQuestion options, Codex `PermissionRequest`/`agent-turn-complete` — a daemon speaks one short, British-butler-toned sentence so you don't miss the moment while pouring coffee or stepping away from the screen. One daemon serves both clients.

```
[ Claude Code asks: Allow `rm -rf /` ? ]
                  │
                  ▼
   "Sir, that command appears rather drastic."
```

The default stack is **fully local, zero-cost**: Ollama for phrasing, CosyVoice 3 for the voice. When the local LLM is unreachable, phrasing falls through a chain of **free** cloud providers (Zhipu GLM, SiliconFlow) before any paid one (DeepSeek) and finally a built-in template — the voice keeps working without surprise bills.

## How it works

```
Claude Code ──Notification / PreToolUse hooks──┐
            └ UserPromptSubmit / PostToolUse ──┤
                                               │
Codex CLI   ──PermissionRequest / PreToolUse ──┤──► jarvis-hook (one-shot, <10ms)
            └ UserPromptSubmit / PostToolUse ──┤
            └ notify (agent-turn-complete) ────┘
                                                │
                                                ▼ Unix socket
                                    jarvis-daemon (launchd, KeepAlive)
                                                │
                          ┌─────────────────────┴─────────────────────┐
                          ▼                                           ▼
                   phrase router                                 TTS engine
              (LLM picks Jarvis line)                       (synthesises audio)
                          │                                           │
     Ollama → SiliconFlow → Zhipu → DeepSeek         CosyVoice 3 → XTTS → say
                                                                      │
                                                                      ▼
                                                                   ffplay / afplay
```

See [`docs/CODEX.md`](docs/CODEX.md) for the Codex CLI event mapping and
verification recipe, or [`docs/SWITCHING.md`](docs/SWITCHING.md) for
swapping providers afterwards.

- Hook is fire-and-forget for notifications (returns under 10ms; never blocks CC).
- Daemon runs forever under launchd, restarted on crash.
- 10-second sliding-window dedup keyed by `(cwd, type, tool)`.
- Bounded queue (drops oldest when >5 events backlogged).
- English / Chinese auto-detect from `CLAUDE.md` / `AGENTS.md` / `README.md` in the event's `cwd`.
- When the local LLM (Ollama) slips onto the cloud fallback, Jarvis announces it audibly so you notice before you start burning credits.
- Optional second job on the same hook + daemon: with the `skills` extra, `UserPromptSubmit` also retrieves and injects the most relevant installed skill per turn (a ~10-40ms daemon round-trip) — see [Skill governance](#skill-governance-rag-over-skills). Off by default; every other event stays fire-and-forget.

## Requirements

- macOS 13+, Apple Silicon (M1/M2/M3/M4). Not tested on Intel.
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).
- Claude Code installed and authenticated.
- At least one **LLM** source:
  - **Ollama** (recommended, local & free) running `qwen3:8b` or similar, OR
  - a **free cloud** key — **Zhipu** GLM-4-Flash or **SiliconFlow** Qwen2.5-7B (both China-direct, OpenAI-compatible), OR
  - **DeepSeek** (cloud, very cheap) / Anthropic / OpenAI API key.
- At least one **TTS** source:
  - **CosyVoice 3** — Apache-2.0, local voice clone, Apple Silicon Metal (`--extra cosyvoice`, recommended default), OR
  - **XTTS-v2** — voice clone, Apple Silicon MPS via PyTorch (`--extra xtts`; weights under CPML / non-commercial), OR
  - **ElevenLabs** API key with `text_to_speech` scope (cloud), OR
  - macOS built-in `say` (zero setup, robotic — also the universal fallback).

## Install

```bash
git clone https://github.com/JobinJia/jarvis-cli.git
cd jarvis

# Pick the TTS path you want; extras are additive.
uv sync --extra cosyvoice          # recommended
# uv sync --extra xtts             # legacy path (CPML non-commercial)
# uv sync --extra cosyvoice --extra xtts   # keep both available
```

Export at least one LLM key into your shell rc **before** running install — it gets baked into the launchd plist so the background daemon can see it:

```bash
echo 'export DEEPSEEK_API_KEY=sk-...'       >> ~/.zshrc   # optional, only if you keep deepseek as fallback
# (optional)
echo 'export ELEVENLABS_API_KEY=sk_...'     >> ~/.zshrc
echo 'export ANTHROPIC_API_KEY=sk-ant-...'  >> ~/.zshrc
echo 'export OPENAI_API_KEY=sk-...'         >> ~/.zshrc
source ~/.zshrc
```

If you're going **fully local** (Ollama + CosyVoice + `say` fallback), no API keys are needed at all.

Then:

```bash
uv run jarvis install
```

This will:

1. Create `~/.jarvis/{voices,models,logs}/`.
2. Write a default `~/.jarvis/config.toml` if absent.
3. Patch `~/.claude/settings.json` to register `Notification`, `PreToolUse`, `UserPromptSubmit`, and `PostToolUse` hooks pointing at the absolute path of `jarvis-hook` in the project venv. If `~/.codex/` exists, also patch `~/.codex/config.toml` with the equivalent Codex lifecycle hooks plus `notify` (sentinel-fenced block, idempotent; see [`docs/CODEX.md`](docs/CODEX.md)).
4. Write `~/Library/LaunchAgents/com.jobin.jarvis.plist` with your API keys embedded. CosyVoice users need `COQUI_TOS_AGREED=1` only if they also enabled the XTTS path (XTTS uses Coqui-TTS internally).
5. `launchctl load` the plist — daemon starts immediately and on every login.

### TTS model setup

**CosyVoice 3** (Apache-2.0, recommended):

```bash
# Download Candle-format weights (~4.7GB on disk)
uv run hf download spensercai/CosyVoice3-0.5B-Candle \
  --local-dir ~/.jarvis/models/cosyvoice3-0.5b-candle

# Provide an English reference clip — 10-30s of clean speech of the voice
# you want cloned (e.g. trimmed from a podcast or interview).
# Save to ~/.jarvis/voices/jarvis_en.wav (mono WAV, ~22050Hz preferred).
```

Add the transcript of that reference clip to `[tts.cosyvoice] ref_text_en` in
`config.toml` — without it, CosyVoice falls back to `inference_cross_lingual`,
which audibly doubles short utterances.

**XTTS-v2** (legacy, CPML non-commercial):

```bash
# Weights auto-download from HuggingFace on the first synthesis call (~2GB).
# Same reference-audio expectation as above.
```

Now **restart any running Claude Code or Codex CLI sessions** so they pick up the patched config files.

> **Upgrading from an older install?** Re-run `uv run jarvis install` to
> register the new `UserPromptSubmit` and `PostToolUse` hooks that drive
> the "stop voice when I respond" behavior.

## Verify

```bash
uv run jarvis status
# {
#   "queue_size": 0,
#   "queue_capacity": 5,
#   "dropped": 0,
#   "last_text": null
# }
```

Fire a synthetic event and listen:

```bash
uv run jarvis test --event permission_prompt --tool Bash
# you should hear a sentence within ~5-15 seconds the first time
# (model load), and ~3-5s on every call after that
```

Trigger the real hook end-to-end:

```
# in any project, open Claude Code and ask it to do
# something that isn't on your auto-allow list, e.g.:
#   "please run sudo ls /root"
# when the approval dialog appears in CC, you should hear Jarvis.
```

## Configuration

Everything lives in `~/.jarvis/config.toml`. The defaults you get after `install`:

```toml
[llm]
provider = "ollama"            # local, zero-cost
# Free-first fallback chain; each link tried in order, then a built-in template.
# SiliconFlow leads: its free tier is quota-based, not load-throttled like Zhipu.
fallbacks = ["siliconflow", "zhipu", "deepseek"]

[llm.ollama]
base_url = "http://localhost:11434"
model = "qwen3:8b"
timeout_seconds = 30

[llm.zhipu]                    # free GLM-4-Flash (real-name verified)
api_key = "..."                # inline key, or set ZHIPU_API_KEY
model = "glm-4-flash"
base_url = "https://open.bigmodel.cn/api/paas/v4"

[llm.siliconflow]              # free Qwen2.5-7B; quota-based, rarely throttled
api_key = "..."                # inline key, or set SILICONFLOW_API_KEY
model = "Qwen/Qwen2.5-7B-Instruct"
base_url = "https://api.siliconflow.cn"

[llm.deepseek]                 # paid but cheap; last resort before the template
api_key = "..."                # inline key, or set DEEPSEEK_API_KEY
model = "deepseek-chat"

[tts]
provider = "cosyvoice"         # Apache-2.0 local voice clone
fallback = "say"               # macOS built-in, universal safety net

[tts.cosyvoice]
model_dir   = "~/.jarvis/models/cosyvoice3-0.5b-candle"
ref_audio_zh = "~/.jarvis/voices/jarvis_zh.wav"
ref_audio_en = "~/.jarvis/voices/jarvis_en.wav"
ref_text_en = ""               # transcript of ref_audio_en — strongly recommended
n_timesteps = 10               # CFM sampling steps (10 = library default)

[tts.xtts]                     # used only if [tts] provider = "xtts"
model_dir   = "~/.jarvis/models/xtts-v2"
ref_audio_zh = "~/.jarvis/voices/jarvis_zh.wav"
ref_audio_en = "~/.jarvis/voices/jarvis_en.wav"
device = "mps"                 # mps | cpu
temperature = 0.5              # < 0.75 default → stable pacing across takes
speed_short = 1.30             # < 60 chars: nudge faster (XTTS slows short lines)
speed_long  = 1.00             # ≥ 60 chars: leave alone (XTTS already flows fast)
short_threshold_chars = 60

[tts.elevenlabs]
api_key_env = "ELEVENLABS_API_KEY"
voice_id = ""                  # set this if you use ElevenLabs!
model = "eleven_turbo_v2_5"

[behavior]
dedup_window_seconds = 10
queue_max_size = 5
voice_language = "en"          # en | zh | auto
events = ["permission_prompt", "idle_prompt", "elicitation_dialog", "ask_user_question"]
phrase_target_chars = 70
phrase_hard_cap = 120
cancel_on_user_action = true   # stop playback when you respond in the originating CC session
mute_subagent_events = true    # subagents work silently; SubagentStart still announces the dispatch

[behavior.privacy]
cloud_redaction = true         # scrub HOME path + secret-shaped tokens before send
```

After editing, reload the daemon to pick up changes:

```bash
launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis.plist
launchctl load   ~/Library/LaunchAgents/com.jobin.jarvis.plist
```

### Recommended profile (zero-cost, OSS-friendly)

This is the default. Local Ollama for phrasing, local CosyVoice 3 for voice — both Apache-2.0, no API calls in steady state.

```toml
[llm]
provider = "ollama"
fallbacks = ["siliconflow", "zhipu", "deepseek"]   # only when Ollama is down; free clouds first, Jarvis announces it

[tts]
provider = "cosyvoice"
fallback = "say"
```

### Cloud-cheap profile

```toml
[llm]
provider = "deepseek"          # cheap and fast TTFT
fallback = "ollama"

[tts]
provider = "elevenlabs"
fallback = "say"

[tts.elevenlabs]
voice_id = "JBFqnCBsd6RMkjVDRZzb"  # George — British narrator, very Jarvis
```

Browse more voices in the [ElevenLabs Voice Library](https://elevenlabs.io/app/voice-library) — copy any voice's ID into `voice_id`. Your EL key only needs `text_to_speech` scope.

### Pure local airplane-mode profile

```toml
[llm]
provider = "ollama"
fallback = ""

[tts]
provider = "say"               # macOS built-in
fallback = ""
```

No network calls of any kind. Voice quality drops; this is your true offline floor.

## Operating

| Action | Command |
|---|---|
| Check daemon health | `uv run jarvis status` |
| Fire a synthetic event | `uv run jarvis test --event permission_prompt --tool Bash` |
| Manually trigger Jarvis (LLM phrases it) | `uv run jarvis say --reason user-input-requested` |
| Manually trigger Jarvis (read exact text) | `uv run jarvis say --text "Sir, shall we proceed?"` |
| Tail daemon logs | `tail -f ~/.jarvis/daemon.log` |
| Reload daemon | `launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis.plist && launchctl load ~/Library/LaunchAgents/com.jobin.jarvis.plist` |
| Update API keys in plist | re-run `uv run jarvis install` (idempotent) |
| Uninstall (keep data) | `uv run jarvis uninstall` |
| Uninstall (wipe data) | `uv run jarvis uninstall --purge` |

## Troubleshooting

**No sound at all.**

- `uv run jarvis status` — daemon reachable?
- `launchctl list | grep jarvis` — service running?
- `tail ~/.jarvis/daemon.log` — error lines?
- Test the leaf: `say "test"` — speakers working?

**Daemon up but `last_text` never changes.** The hook isn't reaching the socket. Common causes:

- You added API keys **after** installing — re-run `jarvis install` to re-bake them into the plist, then reload the daemon.
- Your Claude Code session was running **before** install — restart CC so it re-reads `~/.claude/settings.json`.
- `cat ~/.claude/settings.json | jq '.hooks.Notification'` should show the absolute path to `.venv/bin/jarvis-hook`. If it shows a bare `jarvis-hook`, re-run install.

**You hear "Sir, the local language model … appears unreachable. I am falling back to the cloud."** Ollama either isn't running, the model isn't pulled, or the request timed out. Start `ollama serve`, confirm `ollama list` includes the model from `config.toml`, and try `curl http://localhost:11434/api/tags`. The alert is throttled to once every five minutes during a sustained outage.

**CosyVoice doubles short lines ("Sir Sir, ready ready").** You haven't filled in `[tts.cosyvoice] ref_text_en` — without a transcript the provider falls back to `inference_cross_lingual`, which hallucinates repeats on short utterances. Transcribe your `jarvis_en.wav` (`uvx --from openai-whisper whisper jarvis_en.wav --model tiny --language English`) and paste the cleaned text into the config field.

**XTTS pipeline crashes with `isin_mps_friendly` ImportError.** `transformers>=5` removed the symbol coqui-tts 0.27 still imports. The `[xtts]` extra in `pyproject.toml` pins `transformers<5` precisely for this — re-run `uv sync --extra xtts`.

**`say` reports `Opening output file failed: fmt?`.** That's the macOS `say` binary refusing to write a `.wav` without an explicit `--data-format`. The provider handles this for you (`--data-format=LEF32@22050`); the message means you're running an older copy of the daemon. Re-`uv sync` and reload.

**ElevenLabs 401 with `quota_exceeded`.** Your free credits are out. ElevenLabs returns 401 (not 402/429) for quota — the daemon translates this into a single readable line in `daemon.log` (`TTS provider elevenlabs failed: ElevenLabs quota exhausted: …`). Top up, switch to a key with quota, or move to CosyVoice / XTTS.

**Ollama returns empty text on qwen3 / R1-style models.** Make sure your Ollama is 0.9+; the provider passes `think: false` automatically. If you pinned an older Ollama, upgrade.

**Jarvis says the wrong thing about my command.** Content-awareness pipes `tool_input` (e.g. the actual Bash command, the file basename) into the LLM prompt. If the line still feels generic, check `daemon.log` for whether the provider call succeeded — when LLMs error out, the daemon falls back to the generic template.

## Manual triggers

Claude Code only fires its Notification hook for tool-permission prompts, idle waits, and MCP elicitation. Some scenarios fall outside that — most notably assistant-initiated questions (`AskUserQuestion`, which goes through the `PreToolUse` hook now). Two modes:

**LLM phrases it** — give the model a context label, let it write the line:

```bash
uv run jarvis say --reason "user-input-requested"
# heard: "Sir, your input is awaited."
```

**Speak this exact text** — bypass the LLM entirely (faster, predictable, ideal for reading out the actual question):

```bash
uv run jarvis say --text "Sir, shall this repository be made public or private?"
# heard: <verbatim>
# default --lang en; use --lang zh to switch voice/pronunciation
```

**Override the voice for one call** — useful for A/B-testing candidate voices without editing config:

```bash
uv run jarvis say \
  --text "Sir, sample line for voice tasting." \
  --voice onwK4e9ZLuTAKqWW03F9        # Daniel, a deeper British male
# next `say` without --voice goes back to the config-default voice
```

`--voice` is an ElevenLabs `voice_id` when the active TTS provider is ElevenLabs, or a macOS `say` voice name (eg `Karen`, `Daniel`, `Tingting`) when the active provider is `say`. CosyVoice and XTTS both ignore the override — they clone from the reference audio, not from a named voice.

All modes piggyback on the `idle_prompt` event with a unique `tool_name` (from `--reason` or an auto-uuid) so dedup never collapses successive calls.

## Skill governance (RAG-over-skills)

As you install more Claude Code / Codex skills, every skill's `description` is loaded into the startup prompt whether or not you use it — context grows with your skill count. The `skills` extra hides that long tail and surfaces the right skill *per turn* instead: the `UserPromptSubmit` hook embeds your prompt, retrieves the closest skills from a local index, and injects the matching skill body as `additionalContext`. Because it injects the body directly (not via the Skill tool), it works even for skills hidden from the startup list or living in a disabled plugin.

Opt-in and self-contained — TTS-only users pull none of the embedding stack.

```bash
uv sync --extra skills          # adds fastembed (ONNX, no PyTorch) + numpy + pyyaml

# enable it in ~/.jarvis/config.toml
# [skills]
# enabled = true

# pre-fetch the model (resumable; recommended on slow networks)
jarvis skills download

jarvis skills status        # list discovered skills (no model load)
jarvis skills query 帮我提交代码   # see what a prompt would retrieve

# apply the hiding policy in one shot, reversibly
jarvis skills govern --dry-run   # preview what would be hidden / disabled
jarvis skills govern             # hide standalone skills + disable skill-plugins
jarvis skills govern-status      # what governance currently manages
jarvis skills restore            # undo it from the manifest
```

How it composes with the hook the daemon already runs:

- **Embedding model** — `jinaai/jina-embeddings-v2-base-zh` (bilingual zh/en, ONNX, ~0.64GB, downloaded once into `~/.jarvis/skills/models`). Chosen for cross-lingual recall: a Chinese prompt matches an English skill description. Warm query is ~10-15ms; the model is pre-warmed at daemon start.
- **Retrieval** — cosine similarity over a local index (`catalog.json` + `vectors.npy`), plus a small lexical boost so a shared proper noun (a prompt naming `vercel`/`vue`/`git`) lifts the obvious match. Tiered by score: a confident hit injects the skill body; a weaker one offers a one-line menu; below that, nothing.
- **Hiding the long tail** — `jarvis skills govern` codifies the policy and records a manifest so `skills restore` reverses it exactly. Standalone `~/.claude/skills/` get `skillOverrides` (`"user-invocable-only"`) in `.claude/settings.local.json` — dropped from the model's startup context while `/name` still works. Plugin skills can't be hidden per-skill, so a skill-providing plugin is disabled wholesale (its agents are re-homed to `~/.claude/agents/` first, so e.g. superpowers' `code-reviewer` survives); non-skill plugins are left alone. Either way the retrieval hook still surfaces every skill from disk regardless of enabled state. `--keep name1,name2` leaves a hot-set visible.

Tune thresholds and model under `[skills]` in `config.toml` (see `SkillsConfig`). Everything degrades to a no-op without the extra: no model, no injection, TTS unaffected.

## Project layout

```
src/jarvis/
├── hook_client.py        # one-shot stdin → socket bridge
├── daemon/
│   ├── main.py           # asyncio entrypoint
│   ├── listener.py       # unix-socket server
│   ├── dedup.py          # sliding-window dedup
│   ├── queue.py          # bounded drop-oldest queue
│   └── health.py         # /health on 127.0.0.1:9527
├── phrase/
│   ├── router.py         # LLM chain + on_primary_fallback alert hook
│   ├── language.py       # cwd → 'zh' | 'en'
│   ├── prompt.py         # Jarvis-tone system prompt + few-shot
│   ├── templates.py      # final fallback strings
│   └── providers/        # deepseek, anthropic, openai, ollama
├── tts/
│   ├── engine.py         # primary → fallback
│   └── providers/        # cosyvoice, xtts, elevenlabs, say
├── skills/               # RAG-over-skills (optional `skills` extra)
│   ├── catalog.py        # scan CC+Codex+plugin SKILL.md
│   ├── embedder.py       # fastembed ONNX (lazy)
│   ├── index.py          # catalog.json + vectors.npy
│   ├── retriever.py      # cosine + lexical boost
│   ├── injector.py       # tiered body / menu / none
│   ├── service.py        # daemon-side query + per-session dedup
│   ├── govern.py         # apply/restore the hiding policy (manifest)
│   └── cli.py            # jarvis skills status|query|download|govern|restore
├── player.py             # afplay + ffplay (streaming) wrappers
├── config.py             # TOML loader, dataclass schema
└── install.py            # CLI: install / uninstall / status / test / say
```

Further reading lives under [`docs/`](docs/):
- [`docs/CODEX.md`](docs/CODEX.md) — Codex CLI event mapping, auto-patch internals, verification recipe.
- [`docs/SWITCHING.md`](docs/SWITCHING.md) — provider/profile swap recipes (XTTS ⇄ CosyVoice ⇄ ElevenLabs ⇄ `say`).

321+ unit + integration tests under `tests/`. Run with `uv run pytest`.

## Releasing

Releases are cut by pushing a `v*` tag. The [`release` workflow](.github/workflows/release.yml)
verifies the tag matches the `version` in `pyproject.toml`, builds the sdist +
wheel with `uv build`, and publishes a GitHub Release with the artifacts and
auto-generated notes attached.

There is no PyPI publish: the `cosyvoice` extra installs its wheel via a direct
URL (`allow-direct-references`), which PyPI rejects. Distribution is the source
tree plus the artifacts attached to each GitHub Release.

```bash
# bump the version, keep the lockfile in sync, commit
$EDITOR pyproject.toml                       # e.g. 0.4.0 → 0.4.1
uv lock
git commit -am "chore(release): bump 0.4.0 → 0.4.1"

# tag and push — the workflow builds and publishes the release
git tag -a v0.4.1 -m "v0.4.1 — <highlights>"
git push origin main v0.4.1
```

The tag/version guard fails the build when the two disagree, so a mistagged
release never ships.

## License

The project code is MIT — see `LICENSE`.

Third-party model weights have their own licenses; the project leaves the user
in control of which path they install:

- **CosyVoice 3** (`spensercai/CosyVoice3-0.5B-Candle`, `cosyvoice3.rs`) — **Apache-2.0**. Commercial use OK.
- **XTTS-v2** (`coqui/XTTS-v2`) — **CPML, non-commercial**. The model weights themselves prohibit commercial use; the `[xtts]` extra is kept available for personal / research deployments only.
- **ElevenLabs / DeepSeek / Anthropic / OpenAI** — bound by each provider's own ToS.

Voice samples, recorded models, and synthesised audio are subject to their own terms — never commit reference audio or generated voice clones of real persons to this repo.
