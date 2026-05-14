# jarvis-cc

A Jarvis-voiced notification layer for [Claude Code](https://claude.com/claude-code).

When Claude Code needs your attention — permission prompts, idle waits, MCP elicitation dialogs — a daemon speaks one short, British-butler-toned sentence so you don't miss the moment while pouring coffee or stepping away from the screen.

```
[ Claude Code asks: Allow `rm -rf /` ? ]
                  │
                  ▼
   "Sir, that command appears rather drastic."
```

## How it works

```
Claude Code ──Notification hook──► jarvis-cc-hook (one-shot, <10ms)
                                          │
                                          ▼ Unix socket
                              jarvis-cc-daemon (launchd, KeepAlive)
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                               ▼
                   phrase router                     TTS engine
              (LLM picks Jarvis line)          (synthesises audio)
                          │                               │
              Ollama → DeepSeek → ...        ElevenLabs → XTTS → say
                                                          │
                                                          ▼
                                                       afplay
```

- Hook is fire-and-forget (returns under 10ms; never blocks CC).
- Daemon runs forever under launchd, restarted on crash.
- 10-second sliding-window dedup keyed by `(cwd, type, tool)`.
- Bounded queue (drops oldest when >5 events backlogged).
- English / Chinese auto-detect from `CLAUDE.md` / `AGENTS.md` / `README.md` in the event's `cwd`.

## Requirements

- macOS 13+, Apple Silicon (M1/M2/M3/M4). Not tested on Intel.
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).
- Claude Code installed and authenticated.
- At least one LLM source:
  - **Ollama** (recommended, local & free) running `qwen3:8b` or similar, OR
  - **DeepSeek** API key (cloud, very cheap), OR
  - Anthropic / OpenAI API key.
- At least one TTS source:
  - **ElevenLabs** API key with `text_to_speech` scope (recommended; free tier covers ~10k chars/month), OR
  - macOS built-in `say` (zero setup, robotic), OR
  - XTTS-v2 zero-shot voice cloning (heaviest, needs a 10-30s reference audio file).

## Install

```bash
git clone https://github.com/JobinJia/jarvis-cc.git
cd jarvis-cc
uv sync
```

Export at least one LLM key and one TTS key into your shell rc **before** running install — they get baked into the launchd plist so the background daemon can see them:

```bash
# pick what you have:
echo 'export DEEPSEEK_API_KEY=sk-...'       >> ~/.zshrc
echo 'export ELEVENLABS_API_KEY=sk_...'     >> ~/.zshrc
# (optional)
echo 'export ANTHROPIC_API_KEY=sk-ant-...'  >> ~/.zshrc
echo 'export OPENAI_API_KEY=sk-...'         >> ~/.zshrc
source ~/.zshrc
```

Then:

```bash
uv run jarvis-cc install
```

This will:

1. Create `~/.jarvis-cc/{voices,models,logs}/`.
2. Write a default `~/.jarvis-cc/config.toml` if absent.
3. Patch `~/.claude/settings.json` to register a `Notification` hook pointing at the absolute path of `jarvis-cc-hook` in the project venv.
4. Write `~/Library/LaunchAgents/com.jobin.jarvis-cc.plist` with your API keys embedded.
5. `launchctl load` the plist — daemon starts immediately and on every login.

Now **restart any running Claude Code sessions** so they pick up the patched `settings.json`.

## Verify

```bash
uv run jarvis-cc status
# {
#   "queue_size": 0,
#   "queue_capacity": 5,
#   "dropped": 0,
#   "last_text": null
# }
```

Fire a synthetic event and listen:

```bash
uv run jarvis-cc test --event permission_prompt --tool Bash
# you should hear a sentence within ~1-3 seconds
```

Trigger the real hook end-to-end:

```
# in any project, open Claude Code and ask it to do
# something that isn't on your auto-allow list, e.g.:
#   "please run sudo ls /root"
# when the approval dialog appears in CC,
# you should hear Jarvis within 1-3 seconds.
```

## Configuration

Everything lives in `~/.jarvis-cc/config.toml`. The defaults you get after `install`:

```toml
[llm]
provider = "deepseek"          # primary
fallback = "ollama"            # used if primary times out / errors

[llm.deepseek]
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"

[llm.ollama]
base_url = "http://localhost:11434"
model = "qwen2.5:7b"
timeout_seconds = 10

[tts]
provider = "xtts"              # primary
fallback = "say"               # used if primary fails

[tts.elevenlabs]
api_key_env = "ELEVENLABS_API_KEY"
voice_id = ""                  # set this!
model = "eleven_turbo_v2_5"

[tts.xtts]
model_dir   = "~/.jarvis-cc/models/xtts-v2"
ref_audio_zh = "~/.jarvis-cc/voices/jarvis_zh.wav"
ref_audio_en = "~/.jarvis-cc/voices/jarvis_en.wav"
device = "mps"                 # mps | cpu

[behavior]
dedup_window_seconds = 10
queue_max_size = 5
voice_language = "auto"        # auto | zh | en
events = ["permission_prompt", "idle_prompt", "elicitation_dialog"]
phrase_max_chars = 30
```

After editing, reload the daemon to pick up changes:

```bash
launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis-cc.plist
launchctl load   ~/Library/LaunchAgents/com.jobin.jarvis-cc.plist
```

### Recommended profile (no recording, no GPU, free-ish)

```toml
[llm]
provider = "ollama"            # local, free
fallback = "deepseek"          # cheap cloud backup

[llm.ollama]
model = "qwen3:8b"
timeout_seconds = 30

[tts]
provider = "elevenlabs"
fallback = "say"

[tts.elevenlabs]
api_key_env = "ELEVENLABS_API_KEY"
voice_id = "JBFqnCBsd6RMkjVDRZzb"  # George — British, narrator, very Jarvis
model = "eleven_turbo_v2_5"
```

Browse more voices in the [ElevenLabs Voice Library](https://elevenlabs.io/app/voice-library) — copy any voice's ID into `voice_id`. Your EL API key only needs `text_to_speech` scope.

### Pure local profile (offline-capable)

```toml
[llm]
provider = "ollama"
fallback = ""

[tts]
provider = "say"               # macOS built-in
fallback = ""
```

No network calls. Voice quality drops; this is your "airplane mode".

## Operating

| Action | Command |
|---|---|
| Check daemon health | `uv run jarvis-cc status` |
| Fire a synthetic event | `uv run jarvis-cc test --event permission_prompt --tool Bash` |
| Manually trigger Jarvis | `uv run jarvis-cc say --reason user-input-requested` |
| Tail daemon logs | `tail -f ~/.jarvis-cc/logs/daemon.stderr.log` |
| Reload daemon | `launchctl unload ~/Library/LaunchAgents/com.jobin.jarvis-cc.plist && launchctl load ~/Library/LaunchAgents/com.jobin.jarvis-cc.plist` |
| Update API keys in plist | re-run `uv run jarvis-cc install` (idempotent) |
| Uninstall (keep data) | `uv run jarvis-cc uninstall` |
| Uninstall (wipe data) | `uv run jarvis-cc uninstall --purge` |

## Troubleshooting

**No sound at all.**

- `uv run jarvis-cc status` — daemon reachable?
- `launchctl list | grep jarvis` — service running?
- `tail ~/.jarvis-cc/logs/daemon.stderr.log` — error lines?
- Test the leaf: `say "test"` — speakers working?

**Daemon up but `last_text` never changes.** The hook isn't reaching the socket. Common causes:

- You added `ELEVENLABS_API_KEY` / `DEEPSEEK_API_KEY` **after** installing — re-run `jarvis-cc install` to re-bake them into the plist, then reload the daemon.
- Your Claude Code session was running **before** install — restart CC so it re-reads `~/.claude/settings.json`.
- `cat ~/.claude/settings.json | jq '.hooks.Notification'` should show the absolute path to `.venv/bin/jarvis-cc-hook`. If it shows a bare `jarvis-cc-hook`, re-run install.

**Ollama returns empty text on qwen3 / R1-style models.** Make sure your Ollama is 0.9+; the provider passes `think: false` automatically. If you pinned an older Ollama, upgrade.

**ElevenLabs 401.** Your API key is missing `text_to_speech` scope. Regenerate it in ElevenLabs → Profile → API Keys with permissions = Full (or include `text_to_speech` explicitly).

## Manual triggers

Claude Code only fires its Notification hook for tool-permission prompts, idle waits, and MCP elicitation. Some scenarios fall outside that — most notably assistant-initiated questions (`AskUserQuestion`). For those, the assistant can `Bash`-call:

```bash
uv run jarvis-cc say --reason "user-input-requested"
```

This pushes a synthetic `idle_prompt` event onto the daemon, bypassing dedup (the `--reason` becomes the `tool_name`, making the dedup hash unique per call). The LLM still generates the phrase from context, so it never repeats verbatim.

## Project layout

```
src/jarvis_cc/
├── hook_client.py        # one-shot stdin → socket bridge
├── daemon/
│   ├── main.py           # asyncio entrypoint
│   ├── listener.py       # unix-socket server
│   ├── dedup.py          # sliding-window dedup
│   ├── queue.py          # bounded drop-oldest queue
│   └── health.py         # /health on 127.0.0.1:9527
├── phrase/
│   ├── router.py         # LLM chain: primary → fallback → templates
│   ├── language.py       # cwd → 'zh' | 'en'
│   ├── prompt.py         # Jarvis-tone system prompt + few-shot
│   ├── templates.py      # final fallback strings
│   └── providers/        # deepseek, anthropic, openai, ollama
├── tts/
│   ├── engine.py         # primary → fallback
│   └── providers/        # xtts, elevenlabs, say
├── player.py             # async afplay wrapper
├── config.py             # TOML loader, dataclass schema
└── install.py            # CLI: install / uninstall / status / test
```

63 unit + integration tests under `tests/`. Run with `uv run pytest`.

## License

MIT. See `LICENSE`.

Voice samples, recorded models, and ElevenLabs-generated audio are subject to their own terms — never commit reference audio or generated voice clones of real persons to this repo.
