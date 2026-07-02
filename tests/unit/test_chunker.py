"""Sentence chunker: sentence-end splitting + first-chunk clause split.

Time-to-first-audio used to wait for the LLM to finish the ENTIRE first
sentence.  Jarvis lines often open with a clause ("Sir, the tests have
failed — ..."), so the chunker now splits the FIRST chunk at a clause
boundary (comma/em-dash, Latin and CJK) once ``FIRST_CHUNK_MIN_CHARS`` is
buffered.  After that first yield, behavior is exactly the old one: only
sentence-ending punctuation at ``MIN_CHUNK_CHARS`` triggers a split —
mid-utterance clause splits chop prosody noticeably.
"""
from __future__ import annotations

import pytest

from jarvis_cli.phrase.chunker import chunk_sentences


async def _collect(parts: list[str]) -> list[str]:
    """Feed *parts* as token deltas and collect every yielded chunk."""
    async def _tokens():
        for part in parts:
            yield part

    return [chunk async for chunk in chunk_sentences(_tokens())]


# --- first-chunk clause split ------------------------------------------------


@pytest.mark.asyncio
async def test_first_chunk_splits_at_first_clause_boundary_past_min():
    """The head clause goes to TTS early: the first comma landing at a token
    boundary AT/AFTER 12 buffered chars yields immediately; the tail then
    yields only on sentence end.  ("Sir," at 4 chars is below the floor, so
    the split happens at the second comma.)"""
    chunks = await _collect([
        "Sir", ",", " the tests have failed", ",",
        " three assertions remain in auth", ".",
    ])
    assert chunks == [
        "Sir, the tests have failed,",
        "three assertions remain in auth.",
    ]


@pytest.mark.asyncio
async def test_second_comma_does_not_split_after_first_yield():
    """Only the FIRST chunk may clause-split; a later comma at a token
    boundary is buffered until sentence end, exactly as before."""
    chunks = await _collect([
        "Sir, right away", ",", " I'll fix that", ",", " sir", ".",
    ])
    assert chunks == ["Sir, right away,", "I'll fix that, sir."]


@pytest.mark.asyncio
async def test_short_head_below_min_chars_is_buffered():
    """"Sir," (4 chars) at a token boundary stays below FIRST_CHUNK_MIN_CHARS:
    no split yet — the text is buffered until a later boundary."""
    chunks = await _collect(["Sir", ",", " done and dusted", "."])
    assert chunks == ["Sir, done and dusted."]


@pytest.mark.asyncio
async def test_comma_inside_a_token_does_not_split():
    """Tokens are arbitrary fragments: a comma arriving mid-token ("Sir, the
    tests") never leaves the buffer ending in a comma, so no clause split
    fires there — the whole sentence yields on its period."""
    chunks = await _collect(["Sir, the tests", " passed", "."])
    assert chunks == ["Sir, the tests passed."]


@pytest.mark.asyncio
@pytest.mark.parametrize("delim", ["，", "、"])
async def test_cjk_clause_delimiters_trigger_first_split(delim: str):
    """中文逗号/顿号 at a token boundary also clause-splits the first chunk."""
    chunks = await _collect([
        f"先生{delim}", "所有测试都已经通过了", delim, "共计三十项", "。",
    ])
    assert chunks == [
        f"先生{delim}所有测试都已经通过了{delim}",
        "共计三十项。",
    ]


# --- regression: sentence-end behavior unchanged ------------------------------


@pytest.mark.asyncio
async def test_plain_sentence_without_comma_unchanged():
    """No clause boundary anywhere: a complete sentence yields on its period
    exactly as before the first-chunk clause split existed."""
    chunks = await _collect(["The tests", " have all", " passed", "."])
    assert chunks == ["The tests have all passed."]


@pytest.mark.asyncio
async def test_trailing_text_still_flushed_at_stream_end():
    """Leftover text with no terminal punctuation is flushed when the token
    stream ends — the clause split must not change the flush path."""
    chunks = await _collect(["Sir, compiling now", ",", " almost"])
    assert chunks == ["Sir, compiling now,", "almost"]
