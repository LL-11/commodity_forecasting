from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta

from commodity_ai.backtest import walk_forward_backtest
from commodity_ai.demo import business_days, seed_demo
from commodity_ai.domain import PriceObservation
from commodity_ai.forecasting import DirectForecastModel, RidgeRegressor, mape, quantile
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

    def test_mape_excludes_zero_actuals(self) -> None:
        self.assertEqual(mape([0.0, 2.0, 4.0], [10.0, 1.0, 6.0]), 0.5)

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
        self.assertIn("test_mape", loaded.metrics)
        self.assertIn("fit_mae", loaded.metrics)

    def test_historical_mode_trains_on_bulk_ingested_eia_history(self) -> None:
        repository = MarketRepository()
        observed_at = datetime(2026, 9, 17, 20, tzinfo=UTC)
        repository.add_prices(
            PriceObservation(
                observation_date=day,
                price=2.5 + index * 0.01,
                publication_timestamp=observed_at,
                ingestion_timestamp=observed_at,
                source="EIA API v2",
            )
            for index, day in enumerate(business_days(date(2025, 1, 2), 100))
        )
        service = ForecastService(repository, model_backend="ridge")

        with self.assertRaisesRegex(ValueError, "found 0"):
            service.run_forecast(observed_at, [1], training_mode="point_in_time")

        row = service.run_forecast(observed_at, [1], training_mode="historical")[0]
        self.assertEqual(row.model_version, "ridge-direct-0.2-historical")
        self.assertEqual(row.metrics["historical_training_mode"], 1.0)
        self.assertGreaterEqual(row.metrics["training_rows"], 20)
        self.assertGreaterEqual(row.metrics["walk_forward_retraining_frequency"], 5.0)
        repository.close()

    def test_invalid_training_mode_is_rejected(self) -> None:
        repository = MarketRepository()
        service = ForecastService(repository, model_backend="ridge")
        with self.assertRaisesRegex(ValueError, "training_mode"):
            service.run_forecast(datetime.now(UTC), [1], training_mode="unknown")
        repository.close()

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
