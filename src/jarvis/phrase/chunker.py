"""Sentence chunker: buffer LLM token deltas and yield complete sentences.

Consumes an async stream of token strings and yields accumulated text each
time a sentence-ending punctuation mark is encountered.  A minimum chunk
length prevents tiny fragments (e.g. abbreviations like "Mr.") from being
emitted prematurely.  Any leftover text is flushed when the token stream ends.

NB: an earlier revision additionally split the FIRST chunk at clause
boundaries (comma/em-dash) to start TTS half a sentence earlier.  Field
probes refuted it on both axes: XTTS reads short fragments with a drawn-out
delivery ("Sir, the orchestra waits —" → 3.3s of audio for five words), and
tiny pieces decode disproportionately slowly (measured RTF 3.10 on a 20-char
clause — per-piece prefill dominates), so the "earlier" start actually
arrived later and sounded worse.  Whole sentences only.

The em-dash rewrite below is NOT that revision.  What XTTS drawls is a
DANGLING clause — one still ending on the dash, with no terminal
punctuation to close the prosody.  Rewriting "A — B" into the two whole
sentences "A." and "B" hands it two closed units instead, and closed
units measure fine ("All done, sir." — 14 chars, 0.6s wall, against 4.3s
for the joined line).  prompt._SYSTEM_BASE already asks for two sentences;
this is the deterministic floor for when the model answers with the joined
form regardless, which it still did 3 times in 8 on 2026-08-31.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator

# Sentence-ending punctuation: Latin (. ! ? ;) plus CJK full-width
# equivalents.  The regex matches the punctuation followed by optional
# closing quotes/brackets/whitespace, so "he said." and "done!" both
# trigger a split.
#
# The em-dash is deliberately NOT here: it never ENDS a sentence.  It is
# handled by _DASH_JOIN below, which rewrites rather than merely splits — a
# bare split would hand XTTS the dangling "Sir, he wishes to push to main —"
# that it renders with a long, drawn-out delivery (the user heard "Sir"
# stretched for seconds).
_SENTENCE_END = re.compile(r'[.!?;。！？；]["\')）\]】]?\s*$')

# "clause — clause" joined by a dash, in either the spaced Latin form or the
# unspaced CJK double dash.  Each joined line costs a whole utterance's
# synthesis before the first sound, so when the model ignores the prompt's
# two-sentence instruction, close the first clause here instead.
_DASH_JOIN = re.compile(r'\s*(?:——|—|--)\s*')

# Fragments shorter than this are held back — catches abbreviations like
# "Mr." and numbered lists ("1.") that end in a period but aren't sentence
# boundaries.
#
# Lowered 20 -> 12 on 2026-08-31, when the phrase prompt started asking for a
# short opening sentence followed by the detail (see prompt._SYSTEM_BASE). The
# whole point of that opening is to be tiny — "All done, sir." is 14 chars and
# synthesises in 0.6s against 4.3s for a full dash-joined line — and at 20 the
# floor swallowed it, holding the opening back until the second sentence
# arrived and defeating the split. 12 still holds "Mr." (3) and "1." (2); the
# cost of a rare late split is one delayed sentence, not a wrong one.
MIN_CHUNK_CHARS = 12


async def chunk_sentences(
    tokens: AsyncIterator[str],
    *,
    min_chars: int = MIN_CHUNK_CHARS,
) -> AsyncIterator[str]:
    """Consume *tokens* and yield complete sentences.

    Parameters
    ----------
    tokens:
        An async iterator of string token deltas from an LLM.
    min_chars:
        Minimum accumulated length before a sentence-end punctuation mark
        triggers a yield.  Shorter fragments are buffered until more text
        arrives.
    """
    buf = ""
    async for token in tokens:
        buf += token
        # Only attempt a split when the buffer is long enough and ends with
        # sentence-ending punctuation.
        if len(buf) >= min_chars and _SENTENCE_END.search(buf):
            sentence = buf.strip()
            if sentence:
                yield sentence
            buf = ""
            continue
        # No terminal punctuation yet.  If the model joined two clauses with a
        # dash, close the first one here so TTS can start on it.  Requires text
        # already past the dash: a buffer still ending ON the dash would yield
        # exactly the dangling fragment the docstring warns about.
        m = _DASH_JOIN.search(buf)
        if m and buf[m.end():].strip():
            head = buf[:m.start()].strip()
            if len(head) >= min_chars:
                if not _SENTENCE_END.search(head):
                    head += "."
                yield head
                buf = buf[m.end():]
    # Flush any remaining text when the stream ends.
    tail = buf.strip()
    if tail:
        yield tail
