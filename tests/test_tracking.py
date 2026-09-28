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
    git_commit,
    payload_fingerprint,
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
    module.log_text = Mock()
    return module


class TrackingTests(unittest.TestCase):
    def test_git_commit_uses_configured_container_revision(self) -> None:
        with (
            patch.dict("os.environ", {"COMMODITY_AI_GIT_COMMIT": "abc123"}),
            patch("commodity_ai.tracking.subprocess.run") as run,
        ):
            self.assertEqual(git_commit(), "abc123")
        run.assert_not_called()

    def test_git_commit_is_unknown_when_git_is_unavailable(self) -> None:
        with (
            patch.dict("os.environ", {"COMMODITY_AI_GIT_COMMIT": ""}),
            patch(
                "commodity_ai.tracking.subprocess.run",
                side_effect=FileNotFoundError("git is not installed"),
            ),
        ):
            self.assertEqual(git_commit(), "unknown")

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

    def test_payload_fingerprint_is_stable_for_mapping_order(self) -> None:
        self.assertEqual(
            payload_fingerprint({"b": 2, "a": 1}),
            payload_fingerprint({"a": 1, "b": 2}),
        )

    def test_rolling_window_logs_distinct_evaluation_and_prediction_fingerprints(
        self,
    ) -> None:
        mlflow = fake_mlflow()
        tracker = MLflowTracker(enabled=True)
        predictions = [{"actual": 3.0, "predicted": 2.5}]
        evaluation = [{"origin_date": "2026-01-01", "actual": 3.0}]
        with patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.xgboost": mlflow.xgboost}):
            tracker.log_rolling_window(
                configuration=configuration(),
                x=[[1.0, 2.0]],
                y=[3.0],
                feature_names=["storage", "temperature"],
                model_parameters={"max_depth": 4},
                requested_model_parameters={"max_depth": 4},
                metrics={"test_mae": 0.5, "test_rmse": 0.5, "test_mape": 0.1},
                predictions=predictions,
                evaluation_snapshot=evaluation,
                training_snapshot={"rows": [{"actual": 2.0}]},
                period_details={"window_index": 0},
            )
        rolling_params = mlflow.log_params.call_args_list[-1].args[0]
        self.assertEqual(rolling_params["evaluation_sha256"], payload_fingerprint(evaluation))
        self.assertEqual(rolling_params["predictions_sha256"], payload_fingerprint(predictions))
        self.assertEqual(rolling_params["requested_xgb_max_depth"], 4)
        artifact_paths = [call.args[1] for call in mlflow.log_dict.call_args_list]
        self.assertIn("data/evaluation_snapshot.json", artifact_paths)
        self.assertIn("data/training_snapshot.json", artifact_paths)

    def test_rolling_summary_logs_parent_audit_artifacts(self) -> None:
        mlflow = fake_mlflow()
        tracker = MLflowTracker(enabled=True)
        snapshot = {"rows": [{"origin_date": "2026-01-01", "actual": 3.0}]}
        with patch.dict(sys.modules, {"mlflow": mlflow, "mlflow.xgboost": mlflow.xgboost}):
            tracker.log_rolling_summary(
                parameters={"period_count": 2},
                metrics={"test_mae": 0.5, "test_rmse": 0.5, "test_mape": 0.1},
                summary={"prediction_count": 20},
                period_configuration={"horizon": 20},
                period_metrics_csv="window_index,test_mae\n0,0.5\n",
                predictions_csv="actual,predicted\n3.0,2.5\n",
                feature_names=["storage", "temperature"],
                dataset_snapshot=snapshot,
            )
        parent_params = mlflow.log_params.call_args.args[0]
        self.assertEqual(parent_params["dataset_sha256"], payload_fingerprint(snapshot))
        self.assertIn("git_commit", parent_params)
        artifact_paths = [call.args[1] for call in mlflow.log_dict.call_args_list]
        self.assertIn("config/features.json", artifact_paths)
        self.assertIn("data/historical_dataset.json", artifact_paths)


if __name__ == "__main__":
    unittest.main()
