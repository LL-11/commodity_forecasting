from __future__ import annotations

import importlib.util
import unittest
from datetime import UTC, date, datetime, timedelta
from typing import cast

from commodity_ai.dashboard import (
    build_henry_hub_price_chart,
    prepare_price_chart_data,
)
from commodity_ai.domain import PriceObservation

ALTAIR_AVAILABLE = importlib.util.find_spec("altair") is not None


def price_observation(day: date, price: float) -> PriceObservation:
    published = datetime.combine(day, datetime.min.time(), tzinfo=UTC)
    return PriceObservation(day, price, published, published)


class PriceChartDataTests(unittest.TestCase):
    def test_uses_observation_dates_and_sorts_chronologically(self) -> None:
        prices = [
            price_observation(date(2026, 9, 3), 2.90),
            price_observation(date(2026, 9, 1), 2.80),
            price_observation(date(2026, 9, 2), 2.85),
        ]

        chart_data = prepare_price_chart_data(prices)

        self.assertEqual(
            [row["observation_date"] for row in chart_data],
            [
                "2026-09-01T00:00:00",
                "2026-09-02T00:00:00",
                "2026-09-03T00:00:00",
            ],
        )
        self.assertEqual([row["price"] for row in chart_data], [2.80, 2.85, 2.90])

    def test_limits_to_latest_120_observations_after_sorting(self) -> None:
        start = date(2026, 1, 1)
        prices = [
            price_observation(start + timedelta(days=index), float(index))
            for index in range(125)
        ]
        prices.reverse()

        chart_data = prepare_price_chart_data(prices)

        self.assertEqual(len(chart_data), 120)
        self.assertEqual(chart_data[0]["observation_date"], "2026-01-06T00:00:00")
        self.assertEqual(chart_data[-1]["observation_date"], "2026-05-05T00:00:00")

    def test_preserves_missing_calendar_dates(self) -> None:
        chart_data = prepare_price_chart_data(
            [
                price_observation(date(2026, 9, 11), 2.95),
                price_observation(date(2026, 9, 14), 2.89),
            ]
        )

        self.assertEqual(
            [row["observation_date"] for row in chart_data],
            ["2026-09-11T00:00:00", "2026-09-14T00:00:00"],
        )

    def test_rejects_an_invalid_observation_date(self) -> None:
        observation = price_observation(date(2026, 9, 1), 2.80)
        object.__setattr__(observation, "observation_date", cast(date, None))

        with self.assertRaisesRegex(TypeError, "valid observation date"):
            prepare_price_chart_data([observation])

    def test_rejects_a_non_positive_observation_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            prepare_price_chart_data([], limit=0)


@unittest.skipUnless(ALTAIR_AVAILABLE, "Altair UI dependency is not installed")
class PriceChartSpecificationTests(unittest.TestCase):
    def test_chart_uses_temporal_x_axis_and_date_price_tooltips(self) -> None:
        chart_data = prepare_price_chart_data(
            [price_observation(date(2026, 9, 1), 2.80)]
        )

        specification = build_henry_hub_price_chart(chart_data).to_dict()
        encoding = specification["encoding"]

        self.assertEqual(
            specification["data"]["values"][0]["observation_date"],
            "2026-09-01T00:00:00",
        )
        self.assertEqual(encoding["x"]["field"], "observation_date")
        self.assertEqual(encoding["x"]["type"], "temporal")
        self.assertEqual(encoding["x"]["title"], "Date")
        self.assertEqual(encoding["x"]["axis"]["format"], "%Y-%m-%d")
        self.assertEqual(encoding["y"]["field"], "price")
        self.assertEqual(encoding["y"]["title"], "USD/MMBtu")
        self.assertEqual(
            [(tooltip["field"], tooltip["title"]) for tooltip in encoding["tooltip"]],
            [
                ("observation_date", "Observation date"),
                ("price", "Henry Hub ($/MMBtu)"),
            ],
        )
        self.assertEqual(encoding["tooltip"][0]["format"], "%Y-%m-%d")
        self.assertNotIn("index", str(encoding).lower())


if __name__ == "__main__":
    unittest.main()
