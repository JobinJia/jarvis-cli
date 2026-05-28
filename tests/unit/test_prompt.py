from jarvis_cc.phrase.prompt import build_messages
from jarvis_cc.types import Event


def _ev(**kw) -> Event:
    return Event(
        notification_type=kw.get("notification_type", "permission_prompt"),
        tool_name=kw.get("tool_name", "Bash"),
        tool_input=kw.get("tool_input", {}),
    )


def test_build_messages_zh_system_includes_target_and_cap():
    ev = _ev()
    msgs = build_messages(ev, lang="zh", summary="rm -rf /tmp/x",
                          target_chars=70, hard_cap=120)
    assert msgs[0]["role"] == "system"
    assert "J.A.R.V.I.S" in msgs[0]["content"] or "管家" in msgs[0]["content"]
    assert "中文" in msgs[0]["content"]
    assert "70" in msgs[0]["content"]
    assert "120" in msgs[0]["content"]


def test_build_messages_en_swaps_language_clause():
    msgs = build_messages(_ev(), lang="en", summary="",
                          target_chars=70, hard_cap=120)
    assert "English" in msgs[0]["content"]


def test_build_messages_user_blob_contains_summary_not_raw_tool_input():
    ev = _ev(tool_input={"command": "rm -rf /Users/jobin/tmp", "extra": "huge"})
    msgs = build_messages(ev, lang="en", summary="rm -rf ~/tmp",
                          target_chars=70, hard_cap=120)
    last = msgs[-1]["content"]
    assert "rm -rf ~/tmp" in last
    # The full raw tool_input should NOT leak into the prompt
    assert "huge" not in last
    assert "/Users/jobin" not in last


def test_build_messages_includes_tool_name_in_user_blob():
    msgs = build_messages(_ev(tool_name="Write"), lang="en",
                          summary="write c.toml", target_chars=70, hard_cap=120)
    assert "Write" in msgs[-1]["content"]


def test_build_messages_few_shot_present_for_both_langs():
    msgs_zh = build_messages(_ev(), lang="zh", summary="",
                             target_chars=70, hard_cap=120)
    msgs_en = build_messages(_ev(), lang="en", summary="",
                             target_chars=70, hard_cap=120)
    # system + few-shot pairs + final user → at least 5 messages each
    assert len(msgs_zh) >= 5
    assert len(msgs_en) >= 5


def test_build_messages_empty_summary_still_valid():
    msgs = build_messages(_ev(notification_type="idle_prompt", tool_name=None),
                          lang="zh", summary="", target_chars=70, hard_cap=120)
    assert msgs[-1]["role"] == "user"
    last = msgs[-1]["content"]
    assert '"summary": ""' in last or '"summary":""' in last


# --- humor level injection --------------------------------------------------


def test_build_messages_humor_0_renders_deadpan_clause():
    msgs = build_messages(_ev(), lang="en", summary="",
                          target_chars=70, hard_cap=120, humor_level=0)
    sys_msg = msgs[0]["content"]
    assert "deadpan" in sys_msg.lower()
    # Levels 1+ phrasing must NOT be present at level 0.
    assert "MCU" not in sys_msg
    assert "sardonic" not in sys_msg.lower()


def test_build_messages_humor_2_renders_mcu_clause():
    msgs = build_messages(_ev(), lang="en", summary="",
                          target_chars=70, hard_cap=120, humor_level=2)
    sys_msg = msgs[0]["content"]
    assert "MCU" in sys_msg
    assert "banter" in sys_msg.lower()


def test_build_messages_humor_3_renders_sardonic_clause():
    msgs = build_messages(_ev(), lang="en", summary="",
                          target_chars=70, hard_cap=120, humor_level=3)
    sys_msg = msgs[0]["content"]
    assert "sardonic" in sys_msg.lower()


def test_build_messages_out_of_range_humor_clamps():
    """Defensive: a config typo of humor_level=99 must not crash the
    prompt builder. Clamp to the highest defined clause."""
    high = build_messages(_ev(), lang="en", summary="",
                          target_chars=70, hard_cap=120, humor_level=99)
    assert "sardonic" in high[0]["content"].lower()
    low = build_messages(_ev(), lang="en", summary="",
                         target_chars=70, hard_cap=120, humor_level=-5)
    assert "deadpan" in low[0]["content"].lower()


def test_build_messages_default_humor_level_is_light_wit():
    """Backwards compat: callers that don't pass humor_level get the
    same tone they did before this feature existed."""
    msgs = build_messages(_ev(), lang="en", summary="",
                          target_chars=70, hard_cap=120)
    sys_msg = msgs[0]["content"]
    assert "hint of dry wit" in sys_msg
