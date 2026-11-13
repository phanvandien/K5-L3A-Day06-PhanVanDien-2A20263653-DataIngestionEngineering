"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract every physical page; keep its 1-based page number and source.

    Pages that fail or yield no text are still returned (with empty text) so the
    caller can log them: ``metadata["extraction_error"]`` holds the error, and
    ``metadata["raw_chars"]`` the number of characters before normalization.
    OCR is not attempted here.
    """
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        metadata: dict = {}
        try:
            raw = pdf_page.extract_text() or ""
        except Exception as error:  # one broken page must not hide the others
            raw = ""
            metadata["extraction_error"] = f"{type(error).__name__}: {error}"
        metadata["raw_chars"] = len(raw)
        pages.append(DocumentPage(document_id, source, number, normalize_text(raw), metadata))
    return pages


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace, keeping line and paragraph breaks.

    NFC keeps Vietnamese diacritics as single code points, so the same word
    always has the same characters in chunks, questions and evidence quotes.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("­", "").replace("​", "").replace("﻿", "")
    text = text.replace(" ", " ")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line = paragraph break
    return text.strip()