from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

from .domain import ReportChunk, utc_now
from .rag_types import cosine_similarity
from .repository import MarketRepository


class EmbeddingProvider(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbeddingProvider:
    """Deterministic local embedding fallback based on signed feature hashing."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("embedding dimensions must be at least 32")
        self.dimensions = dimensions
        self.name = f"hash-{dimensions}-v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        tokens = [token.lower() for token in text.split() if token]
        features = tokens + [f"{left}_{right}" for left, right in pairwise(tokens)]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    """Hosted semantic embeddings; requires OPENAI_API_KEY."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        from openai import OpenAI

        self.model = model
        self.name = f"openai-{model}"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [list(item.embedding) for item in response.data]


def configured_embedding_provider() -> EmbeddingProvider:
    provider = os.getenv("COMMODITY_AI_EMBEDDING_PROVIDER", "hash").lower()
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    if provider != "hash":
        raise ValueError("embedding provider must be 'hash' or 'openai'")
    return HashEmbeddingProvider()


class SQLiteVectorIndex:
    def __init__(self, repository: MarketRepository, provider: EmbeddingProvider) -> None:
        self.repository = repository
        self.provider = provider

    def ensure_indexed(self, chunks: Sequence[ReportChunk]) -> dict[str, list[float]]:
        existing = self.repository.chunk_embeddings(
            self.provider.name, (chunk.chunk_id for chunk in chunks)
        )
        missing = [chunk for chunk in chunks if chunk.chunk_id not in existing]
        if missing:
            vectors = self.provider.embed([chunk.text for chunk in missing])
            new_rows = list(zip((chunk.chunk_id for chunk in missing), vectors, strict=True))
            self.repository.save_chunk_embeddings(self.provider.name, new_rows, utc_now())
            existing.update(dict(new_rows))
        return existing

    def similarities(self, query: str, chunks: Sequence[ReportChunk]) -> dict[str, float]:
        vectors = self.ensure_indexed(chunks)
        query_vector = self.provider.embed([query])[0]
        return {
            chunk.chunk_id: cosine_similarity(query_vector, vectors[chunk.chunk_id])
            for chunk in chunks
        }
