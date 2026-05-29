# Provider switching guide

`jarvis-cli` decouples *what generates the line* (LLM) from *what speaks
it* (TTS). Both sides are configured in `~/.jarvis-cli/config.toml` and
either can be swapped at runtime without touching code — restart the
daemon, that's it.

This guide is the source of truth for the available providers, the
recipes for common switches, and the verification commands.

## At a glance

| Provider | Role | Cost | License | TTFT (warm) | Notes |
|---|---|---|---|---|---|
| **CosyVoice 3** | TTS | $0 | Apache-2.0 | ~10 s | Local voice clone via `cosyvoice3.rs` (Metal). OSS-clean. |
| **XTTS-v2** | TTS | $0 | CPML (non-commercial) | ~5 s | Local voice clone via coqui-tts. Faster than CosyVoice; license limits commercial reuse. |
| **ElevenLabs** | TTS | $$ | Per ToS | ~0.5 s (streaming) | Cloud. Fastest but burns credits. |
| **macOS `say`** | TTS | $0 | System | ~1 s | Built-in. Robotic. The universal fallback. |
| **Ollama** | LLM | $0 | Apache/MIT (per model) | ~1-2 s | Local LLM (recommend `qwen3:8b`). |
| **DeepSeek** | LLM | ~$0 | Per ToS | ~1 s | Cloud, cheapest of the cloud LLMs. |
| **Anthropic / OpenAI** | LLM | $$ | Per ToS | ~1 s | Cloud premium. |

Every TTS supports an independent `fallback` provider that takes over
when the primary errors. `[tts] fallback = "say"` is the recommended
universal safety net.

## How to switch

All switches follow the same two-step recipe:

```bash
# 1. Edit ~/.jarvis-cli/config.toml — change `provider` and/or `fallback`
$EDITOR ~/.jarvis-cli/config.toml

# 2. Reload the daemon so it re-reads the file
launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis-cli.plist
launchctl load   ~/Library/LaunchAgents/com.jobin.jarvis-cli.plist

# Shortcut for the same:
launchctl kickstart -k "gui/$(id -u)/com.jobin.jarvis-cli"
```

The daemon process picks up the change on the next event. The first
event after a TTS-provider switch pays a model-load cost
(~10-30 s depending on provider); steady-state TTFT is what's in the
table above.

## Common recipes

### Local-first, zero-cost (default after install)

```toml
[llm]
provider = "ollama"
fallback = "deepseek"    # only fires when ollama is unreachable;
                         # Jarvis announces the slip out loud

[tts]
provider = "cosyvoice"
fallback = "say"
```

Apache-2.0 model weights all the way down; nothing leaves the machine
in steady state.

### Faster local (legacy XTTS path)

```toml
[tts]
provider = "xtts"
fallback = "say"
```

About 2× faster than CosyVoice 3 (~5 s vs ~10 s TTFT). The catch is
the XTTS-v2 weights are CPML-licensed (non-commercial). Fine for
personal use, awkward for a shared OSS deployment.

### Cloud-cheap (ElevenLabs primary, free fallback)

```toml
[tts]
provider = "elevenlabs"
fallback = "cosyvoice"   # when EL quota runs out, the chain
                         # falls back to local — zero downtime,
                         # the user notices via slower TTFT
[tts.elevenlabs]
voice_id = "JBFqnCBsd6RMkjVDRZzb"   # ElevenLabs "George" — Jarvis-ish British male
```

Sub-second TTFT while EL has credits; gracefully degrades to local
when not.

### Pure offline / airplane mode

```toml
[llm]
provider = "ollama"
fallback = ""            # explicit "no fallback"

[tts]
provider = "say"         # or "cosyvoice" if you've installed the weights
fallback = ""
```

No network calls. `say` voice quality is robotic but works on every Mac.

### Trying a different voice without changing providers

Drop a new mono 22050 Hz WAV at
`~/.jarvis-cli/voices/jarvis_en.wav` (or `…_zh.wav`) — 10-30 s of
clean speech of the voice you want cloned. Update
`[tts.cosyvoice] ref_text_en` to the transcript of that clip
(use `uvx --from openai-whisper whisper jarvis_en.wav --model tiny`
to get a starting transcript).

No need to change `[tts] provider`. Just reload the daemon.

## Verification

```bash
# Is the daemon alive?
uv run jarvis-cli status
# {"queue_size": 0, "queue_capacity": 5, "dropped": 0, "last_text": "..."}

# Fire a real event end-to-end (LLM + TTS)
uv run jarvis-cli test --event permission_prompt --tool Bash

# Manually speak verbatim text (bypasses the LLM; pure TTS round-trip)
uv run jarvis-cli say --text "Sir, the system is now under XTTS."

# Tail the daemon log to see which provider actually ran
tail -f ~/.jarvis-cli/daemon.log
```

After a switch, the first synthesis line in the log identifies the
loaded provider, e.g.:

```
INFO  | jarvis_cli.tts.providers.xtts:_load_model:36 - Loading XTTS-v2 ...
INFO  | jarvis_cli.tts.providers.cosyvoice:_load_model:43 - Loading CosyVoice3 ...
```

## When to pick what

- **Most users, most of the time** → `cosyvoice` + `say` fallback. Free, OSS-clean, good enough TTFT.
- **You're on this machine, personal use, want snappier feel** → `xtts` (still free, ~2× faster) and accept the CPML license caveat.
- **Demo / production where TTFT matters more than money** → `elevenlabs` primary with `cosyvoice` fallback for cost spikes.
- **CI / headless / no GPU** → `say` primary; quality drops but synthesis is instant and zero-config.
- **Working offline** → `ollama` + `say` (or `cosyvoice` if weights are already on disk).

## Troubleshooting after a switch

| Symptom | Likely cause |
|---|---|
| Daemon log shows `Loading <provider>` but nothing plays | First-call model load in progress; wait 10-30 s. |
| `provider X failed` then `provider Y` plays | Primary errored; fallback chain working as designed. Check primary's prereqs. |
| `ElevenLabs quota exhausted` | EL free tier is used up. Top up, swap key, or move primary to `cosyvoice` / `xtts`. |
| `Loading XTTS-v2 ...` then nothing | The `[xtts]` extra wasn't installed: `uv sync --extra xtts`. |
| CosyVoice doubles short lines ("Sir Sir, ready ready") | `[tts.cosyvoice] ref_text_en` is empty. Fill in the transcript and reload. |
| `say` errors `Opening output file failed: fmt?` | Old daemon code. `uv sync` and reload. |

For deeper issues see the Troubleshooting section in the top-level
`README.md`.
