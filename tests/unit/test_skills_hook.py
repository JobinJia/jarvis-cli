import json

from jarvis import hook_client
from jarvis.config import Config


def _cfg(enabled: bool) -> Config:
    cfg = Config()
    cfg.skills.enabled = enabled
    cfg.paths.socket = "/tmp/does-not-exist.sock"
    return cfg


def test_prompt_text_prefers_prompt_field():
    assert hook_client._prompt_text({"prompt": "hello"}) == "hello"
    assert hook_client._prompt_text({"user_prompt": "hi"}) == "hi"
    assert hook_client._prompt_text({"nothing": 1}) == ""


def test_emit_additional_context_envelope(capsys):
    hook_client._emit_additional_context("CTX")
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert payload["hookSpecificOutput"]["additionalContext"] == "CTX"


def test_disabled_skills_is_noop(capsys, monkeypatch):
    called = False

    def _fake(*a, **k):
        nonlocal called
        called = True
        return {"context": "X"}

    monkeypatch.setattr(hook_client, "_request_reply", _fake)
    raw = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "p"})
    hook_client.maybe_inject_skills(raw, _cfg(enabled=False))
    assert called is False
    assert capsys.readouterr().out == ""


def test_user_prompt_submit_injects_context(capsys, monkeypatch):
    monkeypatch.setattr(
        hook_client, "_request_reply",
        lambda sock, payload, t: {"context": "DO THE THING", "mode": "body"},
    )
    raw = json.dumps(
        {"hook_event_name": "UserPromptSubmit", "prompt": "commit my code",
         "session_id": "s1"}
    )
    hook_client.maybe_inject_skills(raw, _cfg(enabled=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["hookSpecificOutput"]["additionalContext"] == "DO THE THING"


def test_no_match_injects_nothing(capsys, monkeypatch):
    monkeypatch.setattr(
        hook_client, "_request_reply",
        lambda sock, payload, t: {"context": None, "mode": "none"},
    )
    raw = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "x",
                      "session_id": "s1"})
    hook_client.maybe_inject_skills(raw, _cfg(enabled=True))
    assert capsys.readouterr().out == ""


def test_empty_prompt_skips_query(capsys, monkeypatch):
    seen = []
    monkeypatch.setattr(
        hook_client, "_request_reply",
        lambda sock, payload, t: seen.append(payload) or {"context": "x"},
    )
    raw = json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "   "})
    hook_client.maybe_inject_skills(raw, _cfg(enabled=True))
    assert seen == []
    assert capsys.readouterr().out == ""


def test_session_start_triggers_refresh(monkeypatch):
    cmds = []
    monkeypatch.setattr(
        hook_client, "_request_reply",
        lambda sock, payload, t: cmds.append(payload.get("command")) or {},
    )
    raw = json.dumps({"hook_event_name": "SessionStart", "source": "startup"})
    hook_client.maybe_inject_skills(raw, _cfg(enabled=True))
    assert cmds == ["skill_refresh"]
