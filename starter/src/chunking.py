"""Fixed-size and structure-aware chunking.

Both functions chunk one page at a time, so every chunk maps to exactly one
source page. Chunk IDs are derived from document, page, parameters and order,
so re-ingesting unchanged text produces the same IDs (no duplicates).
"""
import re

from .models import Chunk, DocumentPage

_SENTENCE_END = re.compile(r"(?<=[.!?…:;])\s+")


def _make_chunk(page: DocumentPage, label: str, index: int, start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"{page.document_id}:p{page.page:04d}:{label}:{index:03d}",
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=page.text[start:end],
        metadata={**page.metadata, "char_start": start, "char_end": end},
    )


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it has no leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping character windows and retain page metadata."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("fixed_size needs size > 0 and 0 <= overlap < size")
    text, label = page.text, f"fixed{size}-{overlap}"
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        s, e = _trim(text, start, min(start + size, len(text)))
        if s < e:
            chunks.append(_make_chunk(page, label, len(chunks), s, e))
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def _spans(text: str, pattern: re.Pattern) -> list[tuple[int, int]]:
    """Spans of the pieces between separator matches, whitespace-trimmed."""
    spans, start = [], 0
    for match in pattern.finditer(text):
        spans.append(_trim(text, start, match.start()))
        start = match.end()
    spans.append(_trim(text, start, len(text)))
    return [(s, e) for s, e in spans if s < e]


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to ``max_size`` characters.

    Paragraphs are separated by blank lines. A paragraph longer than
    ``max_size`` is split at sentence ends; a sentence that is still too long
    is cut into ``max_size`` windows. Units are then packed greedily, so a
    chunk never ends in the middle of a sentence unless that sentence alone
    exceeds the limit.
    """
    if max_size <= 0:
        raise ValueError("by_paragraph needs max_size > 0")
    text = page.text
    units: list[tuple[int, int]] = []
    for p_start, p_end in _spans(text, re.compile(r"\n\s*\n")):
        if p_end - p_start <= max_size:
            units.append((p_start, p_end))
            continue
        for s_start, s_end in _spans(text[p_start:p_end], _SENTENCE_END):
            s_start, s_end = s_start + p_start, s_end + p_start
            while s_end - s_start > max_size:
                units.append(_trim(text, s_start, s_start + max_size))
                s_start += max_size
            if s_start < s_end:
                units.append(_trim(text, s_start, s_end))

    label = f"para{max_size}"
    chunks: list[Chunk] = []
    current: tuple[int, int] | None = None
    for start, end in units:
        if start >= end:
            continue
        if current and end - current[0] <= max_size:
            current = (current[0], end)
            continue
        if current:
            chunks.append(_make_chunk(page, label, len(chunks), *current))
        current = (start, end)
    if current:
        chunks.append(_make_chunk(page, label, len(chunks), *current))
    return chunks