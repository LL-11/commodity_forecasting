from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from commodity_ai.chunking import chunk_words
from commodity_ai.domain import Report
from commodity_ai.rag import MarketIntelligenceRetriever, SearchFilters
from commodity_ai.rag_evaluation import RAGEvaluationCase, evaluate_retriever
from commodity_ai.repository import MarketRepository


class RAGSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = MarketRepository()
        self.cutoff = datetime(2025, 1, 15, 18, tzinfo=UTC)
        storage_text = " ".join(
            ["Natural gas storage injections continued during the week."] * 45
            + ["Working gas remained below the five-year average, leaving a storage deficit."] * 8
        )
        self.repo.add_reports(
            [
                Report(
                    "storage",
                    "Weekly Natural Gas Storage Update",
                    "EIA",
                    self.cutoff,
                    "https://example.test/storage",
                    "weekly-update",
                    storage_text,
                ),
                Report(
                    "weather",
                    "Regional Weather Outlook",
                    "NOAA",
                    self.cutoff - timedelta(hours=2),
                    "https://example.test/weather",
                    "weather-outlook",
                    "Temperatures are forecast below normal with higher heating degree days.",
                    region="Northeast",
                ),
                Report(
                    "future",
                    "Future Storage Update",
                    "EIA",
                    self.cutoff + timedelta(days=1),
                    "https://example.test/future",
                    "weekly-update",
                    "A severe storage deficit emerged after the cutoff.",
                ),
            ]
        )

    def tearDown(self) -> None:
        self.repo.close()

    def test_chunking_is_overlapping_and_stable(self) -> None:
        words = [f"word{i}" for i in range(50)]
        chunks = list(chunk_words(" ".join(words), chunk_size=20, overlap=5))
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].split()[-5:], chunks[1].split()[:5])

    def test_hybrid_search_ranks_relevant_chunk_and_builds_citation(self) -> None:
        results = MarketIntelligenceRetriever(self.repo).search(
            "inventory deficit relative to five year average", self.cutoff, top_k=3
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].report_id, "storage")
        self.assertIn("EIA.", results[0].citation)
        self.assertIn("(chunk ", results[0].citation)
        self.assertGreater(results[0].bm25_score, 0)
        self.assertGreater(results[0].tfidf_score, 0)
        self.assertGreaterEqual(results[0].vector_score, 0)
        self.assertGreater(results[0].rerank_score, 0)
        self.assertNotIn("future", {row.report_id for row in results})

    def test_metadata_filters_apply_before_ranking(self) -> None:
        filters = SearchFilters(region="Northeast", document_type="weather-outlook")
        results = MarketIntelligenceRetriever(self.repo).search(
            "cold heating weather", self.cutoff, filters=filters
        )
        self.assertEqual([row.report_id for row in results], ["weather"])

    def test_report_chunks_persist_all_required_metadata(self) -> None:
        chunks = self.repo.report_chunks_as_of(self.cutoff, publisher="EIA")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.commodity == "Henry Hub Natural Gas" for chunk in chunks))
        self.assertTrue(all(chunk.source_url for chunk in chunks))
        self.assertEqual([chunk.chunk_number for chunk in chunks], list(range(len(chunks))))

    def test_embeddings_are_persisted_and_evaluation_metrics_are_computed(self) -> None:
        retriever = MarketIntelligenceRetriever(self.repo)
        retriever.search("storage deficit", self.cutoff)
        count = self.repo.connection.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
        self.assertGreater(count, 0)
        metrics = evaluate_retriever(
            retriever,
            [RAGEvaluationCase("storage deficit", self.cutoff, ("storage",))],
            top_k=3,
        )
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["citation_presence"], 1.0)


if __name__ == "__main__":
    unittest.main()
