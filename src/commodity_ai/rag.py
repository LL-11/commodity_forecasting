from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from .domain import ReportChunk
from .embeddings import EmbeddingProvider, SQLiteVectorIndex, configured_embedding_provider
from .rag_types import min_max_normalize
from .repository import MarketRepository
from .reranker import EvidenceReranker

TOKEN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
QUERY_EXPANSIONS = {
    "inventory": ("storage", "stocks"),
    "inventories": ("storage", "stocks"),
    "cold": ("weather", "heating", "hdd"),
    "heat": ("weather", "cooling", "cdd"),
    "exports": ("lng", "feedgas", "pipeline"),
    "production": ("supply", "output"),
}


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text.lower()) if token not in STOP_WORDS]


def expand_query(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(QUERY_EXPANSIONS.get(token, ()))
    return expanded


@dataclass(frozen=True)
class SearchFilters:
    commodity: str | None = None
    region: str | None = None
    document_type: str | None = None
    publisher: str | None = None


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    report_id: str
    title: str
    publisher: str
    publication_timestamp: str
    document_type: str
    commodity: str
    region: str
    source_url: str
    chunk_number: int
    score: float
    bm25_score: float
    tfidf_score: float
    vector_score: float
    rerank_score: float
    excerpt: str
    citation: str


def _inverse_document_frequency(document_frequency: int, document_count: int) -> float:
    return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))


def _bm25(
    query: Counter[str],
    document: Counter[str],
    document_length: int,
    average_length: float,
    frequencies: Counter[str],
    document_count: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    score = 0.0
    for term, query_frequency in query.items():
        term_frequency = document[term]
        if not term_frequency:
            continue
        inverse_frequency = _inverse_document_frequency(frequencies[term], document_count)
        denominator = term_frequency + k1 * (1 - b + b * document_length / max(average_length, 1.0))
        score += inverse_frequency * (term_frequency * (k1 + 1) / denominator) * query_frequency
    return score


def _tfidf_cosine(
    query: Counter[str],
    document: Counter[str],
    frequencies: Counter[str],
    document_count: int,
) -> float:
    terms = set(query) | set(document)
    query_weights: dict[str, float] = {}
    document_weights: dict[str, float] = {}
    for term in terms:
        inverse_frequency = math.log((document_count + 1) / (frequencies[term] + 1)) + 1
        query_weights[term] = query[term] * inverse_frequency
        document_weights[term] = document[term] * inverse_frequency
    dot = sum(query_weights[term] * document_weights[term] for term in terms)
    left = math.sqrt(sum(value * value for value in query_weights.values()))
    right = math.sqrt(sum(value * value for value in document_weights.values()))
    return dot / (left * right) if left and right else 0.0


def _citation(chunk: ReportChunk) -> str:
    published = chunk.publication_timestamp.date().isoformat()
    return (
        f"{chunk.publisher}. {chunk.title}. {published}. "
        f"{chunk.source_url} (chunk {chunk.chunk_number})"
    )


class MarketIntelligenceRetriever:
    """Point-in-time hybrid chunk retriever with citation-ready results."""

    def __init__(
        self,
        repository: MarketRepository,
        *,
        bm25_weight: float = 0.65,
        tfidf_weight: float = 0.15,
        vector_weight: float = 0.45,
        max_chunks_per_report: int = 2,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: EvidenceReranker | None = None,
    ) -> None:
        if any(weight < 0 for weight in (bm25_weight, tfidf_weight, vector_weight)):
            raise ValueError("retrieval weights must be non-negative")
        if bm25_weight + tfidf_weight + vector_weight == 0:
            raise ValueError("retrieval weights must be non-negative and not both zero")
        if max_chunks_per_report < 1:
            raise ValueError("max_chunks_per_report must be positive")
        self.repository = repository
        total = bm25_weight + tfidf_weight + vector_weight
        self.bm25_weight = bm25_weight / total
        self.tfidf_weight = tfidf_weight / total
        self.vector_weight = vector_weight / total
        self.max_chunks_per_report = max_chunks_per_report
        self.embedding_provider = embedding_provider or configured_embedding_provider()
        self.vector_index = SQLiteVectorIndex(repository, self.embedding_provider)
        self.reranker = reranker or EvidenceReranker()

    def search(
        self,
        query: str,
        as_of: datetime,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        selected = filters or SearchFilters()
        chunks = self.repository.report_chunks_as_of(
            as_of,
            commodity=selected.commodity,
            region=selected.region,
            document_type=selected.document_type,
            publisher=selected.publisher,
        )
        if not chunks:
            return []

        query_terms = Counter(expand_query(tokenize(query)))
        document_terms = [
            Counter(tokenize(f"{chunk.title} {chunk.title} {chunk.document_type} {chunk.text}"))
            for chunk in chunks
        ]
        document_frequency: Counter[str] = Counter()
        for terms in document_terms:
            document_frequency.update(terms.keys())
        lengths = [sum(terms.values()) for terms in document_terms]
        average_length = sum(lengths) / len(lengths)
        raw_bm25 = [
            _bm25(query_terms, terms, length, average_length, document_frequency, len(chunks))
            for terms, length in zip(document_terms, lengths, strict=True)
        ]
        maximum_bm25 = max(raw_bm25, default=0.0) or 1.0
        vector_by_chunk = self.vector_index.similarities(query, chunks)
        normalized_vectors = min_max_normalize(
            [vector_by_chunk[chunk.chunk_id] for chunk in chunks]
        )
        scored: list[tuple[float, float, float, float, float, ReportChunk]] = []
        for chunk, terms, bm25_score, vector_score in zip(
            chunks, document_terms, raw_bm25, normalized_vectors, strict=True
        ):
            tfidf_score = _tfidf_cosine(query_terms, terms, document_frequency, len(chunks))
            normalized_bm25 = bm25_score / maximum_bm25
            retrieval_score = (
                self.bm25_weight * normalized_bm25
                + self.tfidf_weight * tfidf_score
                + self.vector_weight * vector_score
            )
            rerank_score = self.reranker.score(list(query_terms), chunk)
            score = 0.8 * retrieval_score + 0.2 * rerank_score
            scored.append((score, normalized_bm25, tfidf_score, vector_score, rerank_score, chunk))
        scored.sort(
            key=lambda item: (item[0], item[5].publication_timestamp, -item[5].chunk_number),
            reverse=True,
        )

        results: list[SearchResult] = []
        per_report: Counter[str] = Counter()
        for score, bm25_score, tfidf_score, vector_score, rerank_score, chunk in scored:
            if score <= 0 or per_report[chunk.report_id] >= self.max_chunks_per_report:
                continue
            results.append(
                SearchResult(
                    chunk.chunk_id,
                    chunk.report_id,
                    chunk.title,
                    chunk.publisher,
                    chunk.publication_timestamp.isoformat(),
                    chunk.document_type,
                    chunk.commodity,
                    chunk.region,
                    chunk.source_url,
                    chunk.chunk_number,
                    score,
                    bm25_score,
                    tfidf_score,
                    vector_score,
                    rerank_score,
                    chunk.text[:600].strip(),
                    _citation(chunk),
                )
            )
            per_report[chunk.report_id] += 1
            if len(results) == top_k:
                break
        return results
