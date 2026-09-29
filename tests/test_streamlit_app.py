from __future__ import annotations

import importlib.util
import os
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from commodity_ai.rolling_backtest import (
    HistoricalDataset,
    PeriodEvaluation,
    RollingEvaluation,
    RollingPrediction,
    RollingWindow,
    TrackedRollingResult,
)

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None

if STREAMLIT_AVAILABLE:
    import streamlit as st
    from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def backtest_result(parent_run_id: str | None = None) -> TrackedRollingResult:
    predictions = (
        RollingPrediction(
            window_index=1,
            row_index=0,
            origin_date=date(2026, 1, 2),
            target_date=date(2026, 1, 9),
            origin_cutoff=datetime(2026, 1, 2, tzinfo=UTC),
            target_publication_timestamp=datetime(2026, 1, 10, tzinfo=UTC),
            actual=3.1,
            predicted=3.0,
            naive_baseline=2.9,
            moving_average_baseline=2.8,
            p10=2.7,
            p90=3.3,
            calibration_observations=10,
            calibration_max_target_publication=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        RollingPrediction(
            window_index=1,
            row_index=1,
            origin_date=date(2026, 1, 5),
            target_date=date(2026, 1, 12),
            origin_cutoff=datetime(2026, 1, 5, tzinfo=UTC),
            target_publication_timestamp=datetime(2026, 1, 13, tzinfo=UTC),
            actual=3.2,
            predicted=3.3,
            naive_baseline=3.1,
            moving_average_baseline=3.0,
            p10=2.9,
            p90=3.5,
            calibration_observations=11,
            calibration_max_target_publication=datetime(2026, 1, 2, tzinfo=UTC),
        ),
    )
    metrics = {
        "test_mae": 0.1,
        "test_rmse": 0.1,
        "test_mape": 0.032,
    }
    period = PeriodEvaluation(
        window=RollingWindow(1, 0, 2),
        predictions=predictions,
        metrics=metrics,
        refits=(),
        final_training_indices=(),
        train_start=date(2025, 9, 1),
        train_end=date(2025, 12, 31),
    )
    dataset = HistoricalDataset((), 5, datetime(2026, 1, 13, tzinfo=UTC), "live", {})
    summary = {
        "horizon": 5,
        "period_count": 1,
        "pooled_metrics": metrics,
        "period_statistics": {name: {"mean": value} for name, value in metrics.items()},
    }
    evaluation = RollingEvaluation(dataset, (period,), summary, 0)
    return TrackedRollingResult(evaluation, parent_run_id, ())


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Streamlit UI dependencies are not installed")
class StreamlitBacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()

    def tearDown(self) -> None:
        st.cache_resource.clear()

    def test_model_evaluation_tab_handles_missing_results_with_tracking_disabled(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COMMODITY_AI_DB": ":memory:",
                "COMMODITY_AI_MLFLOW_ENABLED": "false",
            },
        ):
            app = AppTest.from_file(APP_PATH, default_timeout=10).run()

        self.assertEqual(app.exception, [])
        self.assertIn("Model Evaluation", [tab.label for tab in app.tabs])
        self.assertIn(
            "Rolling backtest results are not available for this forecast configuration.",
            [message.value for message in app.info],
        )
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["MLflow tracking"], "Disabled")
        self.assertEqual(metrics["Run ID"], "Not recorded")
        training_mode = next(
            selector for selector in app.selectbox if selector.label == "Training mode"
        )
        self.assertEqual(training_mode.value, "Historical (bulk EIA, non-vintage)")
        self.assertTrue(any("revision-biased" in message.value for message in app.warning))
        self.assertTrue(
            any("not trading or investment advice" in item.value for item in app.caption)
        )

    def test_model_evaluation_tab_renders_results_and_mlflow_trace(self) -> None:
        with patch.dict(
            os.environ,
            {
                "COMMODITY_AI_DB": ":memory:",
                "COMMODITY_AI_MLFLOW_ENABLED": "true",
                "MLFLOW_EXPERIMENT_NAME": "ui-test-experiment",
            },
        ):
            app = AppTest.from_file(APP_PATH, default_timeout=10)
            app.session_state["rolling_backtest_result"] = backtest_result("run-123")
            app.session_state["rolling_backtest_run_name"] = "ui-test-run"
            app.session_state["rolling_backtest_configuration"] = (5, 40, 10, 10)
            app.run()

        self.assertEqual(app.exception, [])
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["MLflow tracking"], "Enabled")
        self.assertEqual(metrics["Experiment"], "ui-test-experiment")
        self.assertEqual(metrics["Run ID"], "run-123")
        self.assertEqual(metrics["Run name"], "ui-test-run")
        self.assertEqual(metrics["Rolling windows"], "1")
        self.assertEqual(metrics["Forecast horizon"], "5 days")
        self.assertEqual(metrics["Aggregate MAE"], "0.100")
        self.assertEqual(metrics["Aggregate RMSE"], "0.100")
        self.assertEqual(metrics["Aggregate MAPE"], "3.2%")
        self.assertIn("Rolling backtest windows", [item.label for item in app.expander])


if __name__ == "__main__":
    unittest.main()
