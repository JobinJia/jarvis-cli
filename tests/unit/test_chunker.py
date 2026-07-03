"""Sentence chunker: whole sentences only.

An earlier revision split the FIRST chunk at clause boundaries
(comma/em-dash) to start TTS half a sentence earlier.  Field probes refuted
it on both axes — XTTS reads short fragments with a drawn-out delivery, and
tiny pieces decode disproportionately slowly (RTF 3.10 measured on a 20-char
clause), so the split made first audio LATER and worse-sounding.  These
tests lock the whole-sentence contract, including the cases that used to
clause-split.
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


@pytest.mark.asyncio
async def test_sentences_yield_only_on_sentence_end():
    chunks = await _collect([
        "Sir", ",", " the tests have failed", ",",
        " three assertions remain in auth", ".",
    ])
    assert chunks == ["Sir, the tests have failed, three assertions remain in auth."]


@pytest.mark.asyncio
async def test_commas_never_split_a_chunk():
    """Clause boundaries (commas at token edges included) must NOT yield —
    short fragments like "Sir, right away," sound drawn-out when synthesized
    standalone and cost more decode time than they save."""
    chunks = await _collect([
        "Sir, right away", ",", " I'll fix that", ",", " sir", ".",
    ])
    assert chunks == ["Sir, right away, I'll fix that, sir."]


@pytest.mark.asyncio
@pytest.mark.parametrize("delim", ["，", "、"])
async def test_cjk_clause_delimiters_do_not_split(delim: str):
    """中文逗号/顿号同样不切块——整句在句号处一次产出。"""
    chunks = await _collect([
        f"先生{delim}", "所有测试都已经通过了", delim, "共计三十项", "。",
    ])
    assert chunks == [f"先生{delim}所有测试都已经通过了{delim}共计三十项。"]


@pytest.mark.asyncio
async def test_multiple_sentences_yield_separately():
    chunks = await _collect([
        "The tests have all passed", ".", " The branch is ready for review", ".",
    ])
    assert chunks == [
        "The tests have all passed.",
        "The branch is ready for review.",
    ]


@pytest.mark.asyncio
async def test_short_sentence_below_min_chars_is_buffered():
    """A fragment ending in '.' below MIN_CHUNK_CHARS (e.g. "Mr.") is held
    until more text arrives — the abbreviation guard."""
    chunks = await _collect(["Mr", ".", " Stark has approved", "."])
    assert chunks == ["Mr. Stark has approved."]


@pytest.mark.asyncio
async def test_trailing_text_flushed_at_stream_end():
    chunks = await _collect(["Sir, compiling now", ",", " almost"])
    assert chunks == ["Sir, compiling now, almost"]
