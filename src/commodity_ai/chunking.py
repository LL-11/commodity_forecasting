from __future__ import annotations

import re
from collections.abc import Iterator

from .domain import Report, ReportChunk

WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def chunk_words(text: str, chunk_size: int = 180, overlap: int = 30) -> Iterator[str]:
    """Yield stable word chunks with overlap and no empty trailing chunk."""
    if chunk_size < 20:
        raise ValueError("chunk_size must be at least 20 words")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = normalize_text(text).split()
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        yield " ".join(chunk)
        if start + chunk_size >= len(words):
            break


def chunk_report(report: Report, chunk_size: int = 180, overlap: int = 30) -> list[ReportChunk]:
    return [
        ReportChunk(
            chunk_id=f"{report.report_id}:{number}",
            report_id=report.report_id,
            title=report.title,
            publisher=report.publisher,
            publication_timestamp=report.publication_timestamp,
            document_type=report.document_type,
            commodity=report.commodity,
            region=report.region,
            source_url=report.url,
            chunk_number=number,
            text=text,
        )
        for number, text in enumerate(chunk_words(report.raw_text, chunk_size, overlap))
    ]
