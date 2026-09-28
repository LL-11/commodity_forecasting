from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from commodity_ai.backtest import point_forecast_metrics
from commodity_ai.demo import seed_demo
from commodity_ai.domain import PriceObservation
from commodity_ai.forecasting import DirectForecastModel
from commodity_ai.repository import MarketRepository
from commodity_ai.rolling_backtest import (
    RollingBacktestService,
    build_historical_dataset,
    eligible_training_indices,
    evaluate_period,
    plan_rolling_windows,
    run_rolling_evaluation,
)
from commodity_ai.tracking import MLflowTracker


def ridge_model(horizon: int) -> DirectForecastModel:
    return DirectForecastModel(horizon)


class RollingBacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = MarketRepository()
        seed_demo(self.repository)
        self.dataset = build_historical_dataset(self.repository, 20)

    def tearDown(self) -> None:
        self.repository.close()

    def test_demo_has_five_complete_disjoint_windows(self) -> None:
        windows, trailing = plan_rolling_windows(self.dataset, 40, 10, 10)
        self.assertEqual(len(self.dataset.rows), 140)
        self.assertEqual(
            [(window.test_start_index, window.test_end_index) for window in windows],
            [(88, 98), (98, 108), (108, 118), (118, 128), (128, 138)],
        )
        self.assertEqual(trailing, 2)

    def test_later_bulk_revisions_do_not_replace_point_in_time_labels(self) -> None:
        late_publication = datetime(2026, 1, 1, tzinfo=UTC)
        original_prices = self.repository.prices_as_of(self.dataset.evaluation_as_of)
        self.repository.add_prices(
            PriceObservation(
                row.observation_date,
                row.price + 10.0,
                late_publication,
                late_publication,
                "later-bulk-load",
            )
            for row in original_prices
        )

        dataset = build_historical_dataset(self.repository, 20, late_publication)
        windows, trailing = plan_rolling_windows(dataset, 40, 10, 10)

        self.assertEqual(dataset.data_kind, "synthetic")
        self.assertEqual(len(windows), 5)
        self.assertEqual(trailing, 2)

    def test_training_labels_and_interval_residuals_are_point_in_time_safe(self) -> None:
        windows, _ = plan_rolling_windows(self.dataset, 40, 10, 10)
        period = evaluate_period(self.dataset, windows[0], 40, ridge_model)
        for refit in period.refits:
            forecast_origin = self.dataset.rows[refit.forecast_origin_index]
            self.assertEqual(len(refit.training_indices), 40)
            for index in refit.training_indices:
                training_row = self.dataset.rows[index]
                self.assertLessEqual(
                    training_row.target_price_index,
                    forecast_origin.origin_price_index,
                )
                self.assertLessEqual(
                    training_row.target_publication_timestamp,
                    forecast_origin.origin_cutoff,
                )
        for prediction in period.predictions:
            self.assertGreaterEqual(prediction.calibration_observations, 10)
            self.assertIsNotNone(prediction.p10)
            self.assertIsNotNone(prediction.p90)
            self.assertIsNotNone(prediction.calibration_max_target_publication)
            calibration_publication = prediction.calibration_max_target_publication
            assert calibration_publication is not None
            self.assertLessEqual(
                calibration_publication,
                prediction.origin_cutoff,
            )

    def test_pooled_metrics_recompute_from_disjoint_predictions(self) -> None:
        result = run_rolling_evaluation(self.dataset, 40, 10, 10, ridge_model)
        pooled = [row for period in result.periods for row in period.predictions]
        recomputed = point_forecast_metrics(
            [row.actual for row in pooled],
            [row.predicted for row in pooled],
            [row.naive_baseline for row in pooled],
            [row.moving_average_baseline for row in pooled],
        )
        summary = result.summary
        self.assertEqual(summary["period_count"], 5)
        self.assertEqual(summary["prediction_count"], 50)
        self.assertEqual(
            sum(int(period.metrics["test_observations"]) for period in result.periods),
            summary["prediction_count"],
        )
        self.assertAlmostEqual(summary["pooled_metrics"]["test_mae"], recomputed["mae"])
        self.assertAlmostEqual(summary["pooled_metrics"]["test_rmse"], recomputed["rmse"])
        self.assertAlmostEqual(
            summary["pooled_metrics"]["skill_vs_naive"],
            recomputed["skill_vs_naive"],
        )
        self.assertIn("p10_pinball_loss", summary["period_statistics"])
        self.assertIn("p90_pinball_loss", summary["period_statistics"])

    def test_synthetic_period_is_reproducible(self) -> None:
        windows, _ = plan_rolling_windows(self.dataset, 40, 10, 10)
        first = evaluate_period(self.dataset, windows[0], 40, ridge_model)
        second = evaluate_period(self.dataset, windows[0], 40, ridge_model)
        self.assertEqual(
            [row.predicted for row in first.predictions],
            [row.predicted for row in second.predictions],
        )
        self.assertEqual(first.metrics, second.metrics)

    def test_invalid_or_oversized_windows_fail_actionably(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 40"):
            plan_rolling_windows(self.dataset, 39, 10, 10)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            plan_rolling_windows(self.dataset, 40, 10, 5)
        with self.assertRaisesRegex(ValueError, "at least 2 complete periods"):
            plan_rolling_windows(self.dataset, 60, 20, 20)

    def test_eligible_training_rows_are_fixed_width(self) -> None:
        windows, _ = plan_rolling_windows(self.dataset, 40, 10, 10)
        indices = eligible_training_indices(self.dataset.rows, windows[0].test_start_index, 40)
        self.assertEqual(len(indices), 40)
        self.assertEqual(indices, tuple(sorted(indices)))

    def test_service_returns_evaluation_when_mlflow_is_disabled(self) -> None:
        service = RollingBacktestService(self.repository, MLflowTracker(enabled=False))
        with patch.object(service, "_model", side_effect=ridge_model):
            result = service.run(
                horizon=20,
                train_window=40,
                test_window=10,
                step=10,
                run_name="disabled-tracking",
            )
        self.assertEqual(result.evaluation.summary["period_count"], 5)
        self.assertIsNone(result.parent_run_id)
        self.assertEqual(result.child_run_ids, ())


if __name__ == "__main__":
    unittest.main()
