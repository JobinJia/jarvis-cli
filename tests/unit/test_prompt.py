from jarvis.phrase.prompt import build_messages
from jarvis.types import Event


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


# --- tool_failure / task_complete tone clauses ------------------------------


def test_build_messages_tool_failure_biases_grave_and_concise():
    msgs = build_messages(
        _ev(notification_type="tool_failure", tool_name="Bash"),
        lang="en", summary="Bash failed: npm test: 3 tests failed",
        target_chars=70, hard_cap=120,
    )
    sys_msg = msgs[0]["content"]
    assert "FAILURE" in sys_msg
    assert "no banter" in sys_msg.lower() or "no jokes" in sys_msg.lower()
    # The error gist must reach the user blob.
    assert "3 tests failed" in msgs[-1]["content"]


def test_build_messages_task_complete_biases_brief():
    msgs = build_messages(
        _ev(notification_type="task_complete", tool_name=None),
        lang="en", summary="", target_chars=70, hard_cap=120,
    )
    sys_msg = msgs[0]["content"]
    assert "COMPLETION" in sys_msg
    assert "brief" in sys_msg.lower()


def test_build_messages_ask_user_question_demands_full_enumeration():
    msgs = build_messages(
        _ev(notification_type="ask_user_question", tool_name="AskUserQuestion"),
        lang="en", summary="ask: Pick a colour | options: Red; Blue; Green",
        target_chars=320, hard_cap=640,
    )
    sys_msg = msgs[0]["content"]
    assert "QUESTION" in sys_msg
    assert "EVERY option" in sys_msg
    # Must explicitly release the model from the one-sentence/length rules —
    # otherwise a small model merges 4 options into a half-sentence paraphrase.
    assert "overrides" in sys_msg
    assert "Pick a colour" in msgs[-1]["content"]


def test_build_messages_failure_and_complete_are_mutually_exclusive_clauses():
    fail = build_messages(
        _ev(notification_type="tool_failure"), lang="en", summary="",
        target_chars=70, hard_cap=120,
    )[0]["content"]
    done = build_messages(
        _ev(notification_type="task_complete", tool_name=None), lang="en",
        summary="", target_chars=70, hard_cap=120,
    )[0]["content"]
    assert "FAILURE" in fail and "COMPLETION" not in fail
    assert "COMPLETION" in done and "FAILURE" not in done


# --- per-level few-shot examples ---------------------------------------------
# The 2026-07-09 probe (qwen3:8b): with one fixed example set, humor 0 and 3
# produced identical output — examples beat the system-prompt clause on small
# models. The few-shots must therefore differ per level.


def _assistant_lines(msgs) -> list[str]:
    return [m["content"] for m in msgs[1:-1] if m["role"] == "assistant"]


def test_few_shot_examples_differ_across_humor_levels():
    per_level = [
        _assistant_lines(build_messages(
            _ev(), lang, "x", target_chars=70, hard_cap=120, humor_level=lvl,
        ))
        for lang in ("en", "zh")
        for lvl in (0, 3)
    ]
    en0, en3, zh0, zh3 = per_level
    assert en0 != en3
    assert zh0 != zh3
    # More than a token difference: most scenarios should be reworded.
    assert sum(a != b for a, b in zip(en0, en3)) >= 6


def test_few_shot_failure_examples_stay_grave_at_max_humor():
    """tool_failure examples must not joke at any level — a wisecrack about a
    failure reads as mockery, and _FAILURE_CLAUSE forbids banter outright."""
    msgs = build_messages(_ev(), "en", "x",
                          target_chars=70, hard_cap=120, humor_level=3)
    failure_reply = next(
        msgs[i + 1]["content"] for i, m in enumerate(msgs)
        if m["role"] == "user" and "tool_failure" in m["content"]
    )
    assert "fail" in failure_reply.lower()
    assert "?" not in failure_reply  # statements, not quips


def test_few_shot_out_of_range_level_clamps():
    hi = _assistant_lines(build_messages(
        _ev(), "en", "x", target_chars=70, hard_cap=120, humor_level=99,
    ))
    three = _assistant_lines(build_messages(
        _ev(), "en", "x", target_chars=70, hard_cap=120, humor_level=3,
    ))
    assert hi == three


# --- configurable address ----------------------------------------------------


def test_address_substituted_in_system_and_few_shots_en():
    msgs = build_messages(_ev(), "en", "x",
                          target_chars=70, hard_cap=120, address="Boss")
    assert "Address the user as 'Boss'" in msgs[0]["content"]
    replies = _assistant_lines(msgs)
    assert all("Sir" not in r and "sir" not in r for r in replies)
    assert any("Boss" in r for r in replies)


def test_address_substituted_in_few_shots_zh():
    msgs = build_messages(_ev(), "zh", "x",
                          target_chars=70, hard_cap=120, address="老板")
    replies = _assistant_lines(msgs)
    assert all("先生" not in r for r in replies)
    assert any("老板" in r for r in replies)


def test_default_address_unchanged():
    msgs = build_messages(_ev(), "en", "x", target_chars=70, hard_cap=120)
    assert "Address the user as 'Sir'" in msgs[0]["content"]
    assert any("Sir" in r for r in _assistant_lines(msgs))
