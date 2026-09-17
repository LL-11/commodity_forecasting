from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from .domain import ReportChunk


class EvidenceReranker:
    """Lightweight deterministic reranker using coverage, title, and phrase signals."""

    def score(self, query_tokens: Sequence[str], chunk: ReportChunk) -> float:
        query = [token.lower() for token in query_tokens]
        text = chunk.text.lower()
        title = chunk.title.lower()
        if not query:
            return 0.0
        counts = Counter(token for token in query if token in text or token in title)
        coverage = len(counts) / len(set(query))
        title_coverage = sum(token in title for token in set(query)) / len(set(query))
        phrase_bonus = 1.0 if " ".join(query) in f"{title} {text}" else 0.0
        return min(1.0, 0.65 * coverage + 0.25 * title_coverage + 0.10 * phrase_bonus)
