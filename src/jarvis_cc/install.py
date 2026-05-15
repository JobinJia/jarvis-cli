"""Operator CLI: install / uninstall / status / test.

Commands:
  jarvis-cc install     - write config, hook into ~/.claude/settings.json, install plist
  jarvis-cc uninstall   - reverse install steps (keeps user data)
  jarvis-cc status      - check daemon health
  jarvis-cc test        - send a synthetic event to the daemon
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import httpx

from .config import DEFAULT_CONFIG_PATH, expanduser

PLIST_LABEL = "com.jobin.jarvis-cc"
PLIST_PATH = expanduser(f"~/Library/LaunchAgents/{PLIST_LABEL}.plist")
CLAUDE_SETTINGS_PATH = expanduser("~/.claude/settings.json")
JARVIS_DIR = expanduser("~/.jarvis-cc")


def _is_our_hook(hook: dict) -> bool:
    """Match any entry whose command basename is jarvis-cc-hook (legacy bare
    name or absolute path from a previous install in a different venv)."""
    return Path(hook.get("command", "")).name == "jarvis-cc-hook"


def merge_claude_settings(existing: dict, hook_command: str) -> dict:
    """Install our Notification hook, replacing any prior jarvis-cc-hook entry."""
    out = copy.deepcopy(existing)
    hooks = out.setdefault("hooks", {})
    notification = hooks.setdefault("Notification", [])
    pruned: list[dict] = []
    for matcher in notification:
        kept = [h for h in matcher.get("hooks", []) if not _is_our_hook(h)]
        if kept:
            pruned.append({**matcher, "hooks": kept})
    pruned.append(
        {"matcher": "", "hooks": [{"type": "command", "command": hook_command}]}
    )
    out["hooks"]["Notification"] = pruned
    return out


def remove_from_claude_settings(existing: dict, hook_command: str) -> dict:
    """Strip our jarvis-cc-hook entries. hook_command kept for signature compat."""
    out = copy.deepcopy(existing)
    notification = out.get("hooks", {}).get("Notification", [])
    filtered = []
    for matcher in notification:
        hooks = [h for h in matcher.get("hooks", []) if not _is_our_hook(h)]
        if hooks:
            filtered.append({**matcher, "hooks": hooks})
    if "hooks" in out:
        out["hooks"]["Notification"] = filtered
    return out


def render_plist(label: str, program: str, log_dir: str, env: dict[str, str]) -> str:
    """Render a launchd plist for the daemon."""
    env_xml = "\n".join(
        f"        <key>{k}</key>\n        <string>{v}</string>"
        for k, v in env.items()
    )
    return textwrap.dedent(
        f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{label}</string>
            <key>ProgramArguments</key>
            <array>
                <string>{program}</string>
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{log_dir}/daemon.stdout.log</string>
            <key>StandardErrorPath</key>
            <string>{log_dir}/daemon.stderr.log</string>
            <key>EnvironmentVariables</key>
            <dict>
{env_xml}
            </dict>
        </dict>
        </plist>
        """
    )


def cmd_install(args: argparse.Namespace) -> int:
    # 1. Make jarvis-cc dir tree
    base = Path(JARVIS_DIR)
    (base / "voices").mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)

    # 2. Write default config if not present
    cfg_path = Path(DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        cfg_path.write_text(_default_config_toml(), encoding="utf-8")
        print(f"  wrote {cfg_path}")
    else:
        print(f"  (kept existing {cfg_path})")

    # 3. Patch ~/.claude/settings.json
    settings_path = Path(CLAUDE_SETTINGS_PATH)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            print(f"!! could not parse {settings_path}, refusing to overwrite", file=sys.stderr)
            return 2
    hook_command = shutil.which("jarvis-cc-hook")
    if not hook_command:
        print("!! jarvis-cc-hook not on PATH — did you run `uv sync`?", file=sys.stderr)
        return 4
    merged = merge_claude_settings(existing, hook_command=hook_command)
    settings_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
    print(f"  patched {settings_path} (hook → {hook_command})")

    # 4. Write plist
    program = shutil.which("jarvis-cc-daemon")
    if not program:
        print("!! jarvis-cc-daemon not on PATH — did you run `uv sync`?", file=sys.stderr)
        return 3
    plist_path = Path(PLIST_PATH)
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    # Harvest API keys from the operator's current env so launchd can see them
    env = {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"}
    harvest_keys = [
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ELEVENLABS_API_KEY",
    ]
    for k in harvest_keys:
        v = os.environ.get(k)
        if v:
            env[k] = v
            print(f"  baked {k} into plist")
    plist_path.write_text(render_plist(PLIST_LABEL, program, str(base / "logs"), env))
    print(f"  wrote {plist_path}")

    # 5. Load plist
    subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
    rc = subprocess.run(["launchctl", "load", str(plist_path)], check=False).returncode
    if rc != 0:
        print(f"!! launchctl load returned {rc}", file=sys.stderr)
        return rc

    print(
        "\nDone. Next steps:\n"
        f"  1. Place reference audio at {base / 'voices/jarvis_zh.wav'} and "
        f"{base / 'voices/jarvis_en.wav'}\n"
        f"  2. If you didn't have DEEPSEEK_API_KEY in your env when you ran install,\n"
        f"     set it now and re-run `jarvis-cc install` (this re-bakes the plist).\n"
        f"  3. Restart Claude Code\n"
        f"  4. Run: jarvis-cc test\n"
    )
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    plist_path = Path(PLIST_PATH)
    if plist_path.exists():
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
        plist_path.unlink()
        print(f"  removed {plist_path}")

    settings_path = Path(CLAUDE_SETTINGS_PATH)
    if settings_path.exists():
        try:
            existing = json.loads(settings_path.read_text())
            pruned = remove_from_claude_settings(existing, hook_command="jarvis-cc-hook")
            settings_path.write_text(json.dumps(pruned, indent=2, ensure_ascii=False) + "\n")
            print(f"  cleaned {settings_path}")
        except json.JSONDecodeError:
            pass

    if args.purge:
        base = Path(JARVIS_DIR)
        if base.exists():
            shutil.rmtree(base)
            print(f"  purged {base}")
    else:
        print(f"  (kept {JARVIS_DIR}; pass --purge to remove)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        r = httpx.get("http://127.0.0.1:9527/health", timeout=1.0)
        print(json.dumps(r.json(), indent=2, ensure_ascii=False))
        return 0
    except httpx.HTTPError as exc:
        print(f"daemon unreachable: {exc}", file=sys.stderr)
        return 1


def cmd_test(args: argparse.Namespace) -> int:
    import socket as _socket

    from .config import load_config

    cfg = load_config(DEFAULT_CONFIG_PATH)
    payload = {
        "notification_type": args.event,
        "tool_name": args.tool,
        "tool_input": {},
        "cwd": os.getcwd(),
        "session_id": "test",
    }
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        s.connect(cfg.paths.socket)
        s.sendall((json.dumps(payload) + "\n").encode())
        print(f"sent {args.event} ({args.tool}) to {cfg.paths.socket}")
        return 0
    except OSError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    finally:
        s.close()


def cmd_say(args: argparse.Namespace) -> int:
    """Manual one-shot trigger for cases CC doesn't cover (e.g. an assistant
    that wants to alert the user before showing an AskUserQuestion UI).

    Sends a synthetic `idle_prompt` event; `--reason` gets stuffed into
    `tool_name` so the dedup hash is unique per call (no 10s collision) and
    so the LLM has a hint about why it's speaking.
    """
    import socket as _socket
    import uuid

    from .config import load_config

    cfg = load_config(DEFAULT_CONFIG_PATH)
    reason = args.reason or f"manual-{uuid.uuid4().hex[:8]}"
    payload: dict = {
        "notification_type": "idle_prompt",
        "tool_name": reason,
        "tool_input": {},
        "cwd": os.getcwd(),
        "session_id": "manual",
    }
    if args.text:
        payload["text"] = args.text
        payload["lang"] = args.lang
    if args.voice:
        payload["voice_id"] = args.voice
    s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    try:
        s.connect(cfg.paths.socket)
        s.sendall((json.dumps(payload) + "\n").encode())
        if args.text:
            print(f"queued say (text={args.text!r}, lang={args.lang})")
        else:
            print(f"queued say (reason={reason!r}, LLM will phrase it)")
        return 0
    except OSError as exc:
        print(f"failed: {exc}", file=sys.stderr)
        return 1
    finally:
        s.close()


def _default_config_toml() -> str:
    return textwrap.dedent(
        """\
        # jarvis-cc config.toml — auto-generated, edit freely
        [llm]
        provider = "deepseek"
        fallback = "ollama"

        [llm.deepseek]
        api_key_env = "DEEPSEEK_API_KEY"
        model = "deepseek-chat"

        [llm.ollama]
        base_url = "http://localhost:11434"
        model = "qwen2.5:7b"

        [tts]
        provider = "xtts"
        fallback = "say"

        [tts.xtts]
        model_dir = "~/.jarvis-cc/models/xtts-v2"
        ref_audio_zh = "~/.jarvis-cc/voices/jarvis_zh.wav"
        ref_audio_en = "~/.jarvis-cc/voices/jarvis_en.wav"
        device = "mps"

        [tts.elevenlabs]
        api_key_env = "ELEVENLABS_API_KEY"
        voice_id = ""
        model = "eleven_turbo_v2_5"

        [behavior]
        dedup_window_seconds = 10
        queue_max_size = 5
        # Language Jarvis SPEAKS in. "en" (default British voice identity),
        # "zh", or "auto" (decide per-event from content).
        voice_language = "en"
        events = ["permission_prompt", "idle_prompt", "elicitation_dialog", "ask_user_question"]
        # phrase_max_chars is deprecated and ignored; use the budget below.
        phrase_target_chars = 70
        phrase_hard_cap = 120

        [behavior.privacy]
        cloud_redaction = true
        """
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="jarvis-cc")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("install").set_defaults(func=cmd_install)
    p_un = sub.add_parser("uninstall")
    p_un.add_argument("--purge", action="store_true", help="also remove ~/.jarvis-cc/")
    p_un.set_defaults(func=cmd_uninstall)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p_test = sub.add_parser("test")
    p_test.add_argument("--event", default="permission_prompt")
    p_test.add_argument("--tool", default="Bash")
    p_test.set_defaults(func=cmd_test)
    p_say = sub.add_parser(
        "say",
        help="manually trigger Jarvis to speak (bypasses dedup; for events CC's "
        "Notification hook doesn't cover, eg AskUserQuestion)",
    )
    p_say.add_argument(
        "--reason",
        default=None,
        help="short label that flows into the LLM prompt as `tool_name`",
    )
    p_say.add_argument(
        "--text",
        default=None,
        help="speak this exact text; skips the LLM entirely",
    )
    p_say.add_argument(
        "--lang",
        default="en",
        choices=["en", "zh"],
        help="language of --text (default: en); ignored without --text",
    )
    p_say.add_argument(
        "--voice",
        default=None,
        help="per-call TTS voice override (eg an ElevenLabs voice_id or a "
        "macOS `say` voice name like Karen); defaults to config",
    )
    p_say.set_defaults(func=cmd_say)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
