"""Sentence chunker: buffer LLM token deltas and yield complete sentences.

Consumes an async stream of token strings and yields accumulated text each
time a sentence-ending punctuation mark is encountered.  A minimum chunk
length prevents tiny fragments (e.g. abbreviations like "Mr.") from being
emitted prematurely.  Any leftover text is flushed when the token stream ends.
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
    # Flush any remaining text when the stream ends.
    tail = buf.strip()
    if tail:
        yield tail
