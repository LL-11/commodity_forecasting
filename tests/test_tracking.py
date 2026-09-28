from __future__ import annotations

import sys
import types
import unittest
from datetime import UTC, date, datetime
from unittest.mock import Mock, patch

from commodity_ai.tracking import (
    ForecastRunConfiguration,
    MLflowTracker,
    forecast_metrics,
    forecast_parameters,
)


def configuration() -> ForecastRunConfiguration:
    return ForecastRunConfiguration(
        target="henry_hub_spot_price_usd_per_mmbtu",
        forecast_horizon=5,
        as_of=datetime(2026, 1, 31, tzinfo=UTC),
        training_start=date(2025, 1, 1),
        training_end=date(2025, 12, 31),
        test_start=date(2025, 10, 1),
        test_end=date(2025, 12, 31),
        model_name="xgboost",
    )


def fake_mlflow() -> types.ModuleType:
    module = types.ModuleType("mlflow")
    xgboost = types.ModuleType("mlflow.xgboost")
    xgboost.autolog = Mock()
    module.xgboost = xgboost
    module.set_tracking_uri = Mock()
    module.set_experiment = Mock()
    module.log_params = Mock()
    module.log_metrics = Mock()
    module.log_dict = Mock()
    return module


class TrackingTests(unittest.TestCase):
    def test_tracking_configuration_uses_configured_uri_and_experiment(self) -> None:
        mlflow = fake_mlflow()
        with patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.xgboost": mlflow.xgboost}):
            MLflowTracker(
                enabled=True,
                tracking_uri="file:///temporary/mlruns",
                experiment_name="test-experiment",
            ).configure()
        mlflow.set_tracking_uri.assert_called_once_with("file:///temporary/mlruns")
        mlflow.set_experiment.assert_called_once_with("test-experiment")
        mlflow.xgboost.autolog.assert_called_once_with(
            log_models=True, log_datasets=False, silent=False
        )

    def test_forecast_parameters_include_periods_and_feature_metadata(self) -> None:
        parameters = forecast_parameters(
            configuration(), [[1.0, 2.0]], [3.0], ["storage", "temperature"]
        )
        self.assertEqual(parameters["forecast_horizon"], 5)
        self.assertEqual(parameters["training_start"], "2025-01-01")
        self.assertEqual(parameters["test_end"], "2025-12-31")
        self.assertEqual(parameters["feature_count"], 2)
        self.assertEqual(parameters["features"], "storage,temperature")

    def test_required_metrics_are_validated(self) -> None:
        values = forecast_metrics({"test_mae": 1, "test_rmse": 2, "test_mape": 0.5})
        self.assertEqual(set(values), {"test_mae", "test_rmse", "test_mape"})
        with self.assertRaisesRegex(ValueError, "test_mape"):
            forecast_metrics({"test_mae": 1, "test_rmse": 2})

    def test_feature_and_experiment_configuration_are_logged_as_artifacts(self) -> None:
        mlflow = fake_mlflow()
        tracker = MLflowTracker(enabled=True)
        with patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.xgboost": mlflow.xgboost}):
            tracker.log_forecast_configuration(
                configuration=configuration(),
                x=[[1.0, 2.0]],
                y=[3.0],
                feature_names=["storage", "temperature"],
                model_parameters={"max_depth": 4},
            )
        artifact_paths = [call.args[1] for call in mlflow.log_dict.call_args_list]
        self.assertEqual(artifact_paths, ["config/features.json", "config/experiment_config.json"])
        self.assertEqual(
            mlflow.log_dict.call_args_list[0].args[0],
            {"features": ["storage", "temperature"]},
        )


if __name__ == "__main__":
    unittest.main()
