from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta

from commodity_ai.domain import PriceObservation, Report
from commodity_ai.rag import MarketIntelligenceRetriever
from commodity_ai.repository import MarketRepository


class PointInTimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = MarketRepository()

    def tearDown(self) -> None:
        self.repo.close()

    def test_late_revision_is_not_visible(self) -> None:
        observed = date(2025, 1, 2)
        early = datetime(2025, 1, 3, tzinfo=UTC)
        late = datetime(2025, 1, 8, tzinfo=UTC)
        self.repo.add_prices(
            [
                PriceObservation(observed, 3.0, early, early),
                PriceObservation(observed, 9.0, late, late),
            ]
        )
        self.assertEqual(self.repo.prices_as_of(early + timedelta(hours=1))[0].price, 3.0)
        self.assertEqual(self.repo.prices_as_of(late + timedelta(hours=1))[0].price, 9.0)

    def test_future_report_is_excluded_before_ranking(self) -> None:
        cutoff = datetime(2025, 1, 15, tzinfo=UTC)
        self.repo.add_reports(
            [
                Report(
                    "old",
                    "Storage update",
                    "EIA",
                    cutoff,
                    "https://example/old",
                    "update",
                    "storage deficit",
                ),
                Report(
                    "future",
                    "Storage update",
                    "EIA",
                    cutoff + timedelta(days=1),
                    "https://example/future",
                    "update",
                    "storage deficit severe",
                ),
            ]
        )
        rows = MarketIntelligenceRetriever(self.repo).search("storage deficit", cutoff)
        self.assertEqual([row.report_id for row in rows], ["old"])


if __name__ == "__main__":
    unittest.main()
