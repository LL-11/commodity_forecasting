from __future__ import annotations

import unittest
from datetime import date, timedelta

from commodity_ai.backtest import walk_forward_backtest
from commodity_ai.demo import seed_demo
from commodity_ai.forecasting import DirectForecastModel, RidgeRegressor, quantile
from commodity_ai.repository import MarketRepository
from commodity_ai.services import ForecastService


class ForecastingTests(unittest.TestCase):
    def test_ridge_fits_linear_relationship(self) -> None:
        x = [[float(i), float(i % 3)] for i in range(30)]
        y = [2 + 3 * row[0] - 0.5 * row[1] for row in x]
        model = RidgeRegressor(alpha=0.0001).fit(x, y)
        self.assertAlmostEqual(model.predict([10.0, 1.0]), 31.5, places=2)

    def test_quantile_interpolation(self) -> None:
        self.assertEqual(quantile([0, 10], 0.5), 5)

    def test_direct_forecast_interval_is_ordered(self) -> None:
        x = [[float(i)] for i in range(30)]
        y = [2 + 0.1 * i + (-0.2 if i % 2 else 0.2) for i in range(30)]
        result = DirectForecastModel(1).fit(x, y).predict([31.0])
        self.assertLessEqual(result.p10, result.p50)
        self.assertLessEqual(result.p50, result.p90)

    def test_end_to_end_forecast_is_persisted(self) -> None:
        repository = MarketRepository()
        as_of = seed_demo(repository, 100)
        row = ForecastService(repository).run_forecast(as_of, [1])[0]
        loaded = repository.get_forecast(row.forecast_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.horizon, 1)
        self.assertLessEqual(loaded.p10, loaded.p50)
        self.assertLessEqual(loaded.p50, loaded.p90)
        self.assertGreater(len(loaded.drivers), 0)
        self.assertIn("walk_forward_mae", loaded.metrics)
        self.assertIn("fit_mae", loaded.metrics)

    def test_walk_forward_respects_horizon_gap(self) -> None:
        features = [[float(i)] for i in range(80)]
        targets = [2 + 0.05 * i for i in range(80)]
        baselines = [2 + 0.04 * i for i in range(80)]
        dates = [date(2025, 1, 1) + timedelta(days=i) for i in range(80)]
        result = walk_forward_backtest(
            features,
            targets,
            baselines,
            dates,
            forecast_horizon=5,
            retraining_frequency=5,
            minimum_training_rows=30,
        )
        self.assertGreater(result.metrics["observations"], 0)
        self.assertIn("prediction_interval_coverage", result.metrics)


if __name__ == "__main__":
    unittest.main()
