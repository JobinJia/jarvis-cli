"""Sentence chunker: buffer LLM token deltas and yield complete sentences.

Consumes an async stream of token strings and yields accumulated text each
time a sentence-ending punctuation mark is encountered.  A minimum chunk
length prevents tiny fragments (e.g. abbreviations like "Mr.") from being
emitted prematurely.  Any leftover text is flushed when the token stream ends.
The very first chunk may additionally split at a clause boundary (comma or
em-dash) so TTS can start speaking the opening clause while the LLM is still
finishing the sentence.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator

# Sentence-ending punctuation: Latin (. ! ? ;) including em-dash, plus CJK
# full-width equivalents.  The regex matches the punctuation followed by
# optional closing quotes/brackets/whitespace, so "he said." and "done!"
# both trigger a split.
_SENTENCE_END = re.compile(r'[.!?;—。！？；]["\')）\]】]?\s*$')

# Fragments shorter than this are held back — catches abbreviations like
# "Mr." and numbered lists ("1.") that end in a period but aren't sentence
# boundaries.
MIN_CHUNK_CHARS = 20

# The FIRST chunk may split at a clause boundary (comma/em-dash — Latin and
# CJK) so TTS can start on the opening clause while the LLM finishes the
# sentence.  Only the first chunk: mid-utterance clause splits chop prosody
# noticeably; the utterance opener is where the latency win lives.
_CLAUSE_END = re.compile(r'[,\uff0c—、]["\')）\]】]?\s*$')
FIRST_CHUNK_MIN_CHARS = 12


async def chunk_sentences(
    tokens: AsyncIterator[str],
    *,
    min_chars: int = MIN_CHUNK_CHARS,
) -> AsyncIterator[str]:
    """Consume *tokens* and yield complete sentences.

    Until the first yield, the buffer may also split at a clause boundary
    (see ``_CLAUSE_END`` / ``FIRST_CHUNK_MIN_CHARS``) to cut time-to-first-
    audio; after that, only sentence-ending punctuation triggers a yield.

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
    yielded_any = False
    async for token in tokens:
        buf += token
        # Only attempt a split when the buffer is long enough and ends with
        # sentence-ending punctuation.  A short complete sentence takes
        # priority over the first-chunk clause split below.
        if len(buf) >= min_chars and _SENTENCE_END.search(buf):
            sentence = buf.strip()
            if sentence:
                yield sentence
                yielded_any = True
            buf = ""
        elif (
            not yielded_any
            and len(buf) >= FIRST_CHUNK_MIN_CHARS
            and _CLAUSE_END.search(buf)
        ):
            # First chunk only: split at a clause boundary so playback can
            # begin before the sentence is complete.
            clause = buf.strip()
            if clause:
                yield clause
                yielded_any = True
            buf = ""
    # Flush any remaining text when the stream ends.
    tail = buf.strip()
    if tail:
        yield tail
