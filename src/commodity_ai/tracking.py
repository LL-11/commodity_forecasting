from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_EXPERIMENT_NAME = "henry-hub-xgboost"


class MLflowTrackingError(RuntimeError):
    """Raised when experiment tracking fails, rather than model execution."""


@dataclass(frozen=True)
class ForecastRunConfiguration:
    target: str
    forecast_horizon: int
    as_of: datetime
    training_start: date
    training_end: date
    test_start: date
    test_end: date
    model_name: str
    training_mode: str = "point_in_time"


def dataset_fingerprint(x: Sequence[Sequence[float]], y: Sequence[float]) -> str:
    payload = json.dumps({"x": x, "y": y}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def payload_fingerprint(payload: Any) -> str:
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def git_commit() -> str:
    configured = os.getenv("COMMODITY_AI_GIT_COMMIT")
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def forecast_parameters(
    configuration: ForecastRunConfiguration,
    x: Sequence[Sequence[float]],
    y: Sequence[float],
    feature_names: Sequence[str],
) -> dict[str, str | int]:
    """Convert forecast metadata into values accepted by ``mlflow.log_params``."""
    parameters: dict[str, str | int] = {
        "target": configuration.target,
        "forecast_horizon": configuration.forecast_horizon,
        "as_of": configuration.as_of.isoformat(),
        "training_start": configuration.training_start.isoformat(),
        "training_end": configuration.training_end.isoformat(),
        "test_start": configuration.test_start.isoformat(),
        "test_end": configuration.test_end.isoformat(),
        "feature_count": len(feature_names),
        "model": configuration.model_name,
        "training_mode": configuration.training_mode,
        "training_rows": len(x),
        "dataset_sha256": dataset_fingerprint(x, y),
        "git_commit": git_commit(),
    }
    compact_features = ",".join(feature_names)
    if len(compact_features) <= 500:
        parameters["features"] = compact_features
    return parameters


def forecast_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize required holdout metrics before logging."""
    required = ("test_mae", "test_rmse", "test_mape")
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"forecast metrics missing required values: {', '.join(missing)}")
    return {name: float(value) for name, value in metrics.items()}


class MLflowTracker:
    def __init__(
        self,
        enabled: bool | None = None,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        configured = os.getenv("COMMODITY_AI_MLFLOW_ENABLED", "false").lower()
        self.enabled = enabled if enabled is not None else configured in {"1", "true", "yes"}
        self.tracking_uri: str = (
            tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        )
        self.experiment_name = experiment_name or os.getenv(
            "MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME
        )

    def configure(self, *, enable_xgboost_autologging: bool = True) -> None:
        if not self.enabled:
            return
        try:
            import mlflow
            import mlflow.xgboost

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            if enable_xgboost_autologging:
                mlflow.xgboost.autolog(log_models=True, log_datasets=False, silent=False)
        except Exception as error:
            raise MLflowTrackingError(
                f"MLflow experiment tracking configuration failed for "
                f"'{self.tracking_uri}': {error}"
            ) from error

    @contextmanager
    def start_run(
        self,
        *,
        run_name: str | None = None,
        model_name: str = "xgboost",
        nested: bool = False,
        tags: Mapping[str, str] | None = None,
    ) -> Iterator[str | None]:
        if not self.enabled:
            yield None
            return
        if nested:
            if model_name == "xgboost":
                try:
                    import mlflow.xgboost

                    mlflow.xgboost.autolog(log_models=True, log_datasets=False, silent=False)
                except Exception as error:
                    raise MLflowTrackingError(
                        f"MLflow experiment tracking configuration failed: {error}"
                    ) from error
        else:
            self.configure(enable_xgboost_autologging=model_name == "xgboost")
        try:
            import mlflow

            active_run = mlflow.start_run(run_name=run_name, nested=nested, tags=dict(tags or {}))
        except Exception as error:
            if model_name == "xgboost":
                with suppress(Exception):
                    mlflow.xgboost.autolog(disable=True)
            raise MLflowTrackingError(
                f"MLflow experiment tracking could not start a run: {error}"
            ) from error
        # ActiveRun records FAILED if forecasting raises. Do not reclassify those
        # forecasting failures as tracking failures here.
        try:
            with active_run as run:
                yield run.info.run_id
        except BaseException:
            if model_name == "xgboost":
                with suppress(Exception):
                    mlflow.xgboost.autolog(disable=True)
            raise
        else:
            if model_name == "xgboost":
                try:
                    mlflow.xgboost.autolog(disable=True)
                except Exception as error:
                    raise MLflowTrackingError(
                        f"MLflow experiment tracking failed to stop autologging: {error}"
                    ) from error

    def log_forecast_configuration(
        self,
        *,
        configuration: ForecastRunConfiguration,
        x: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Sequence[str],
        model_parameters: Mapping[str, Any],
        experiment_details: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            import mlflow

            params = forecast_parameters(configuration, x, y, feature_names)
            configuration_artifact = asdict(configuration)
            configuration_artifact.update(
                {
                    "as_of": configuration.as_of.isoformat(),
                    "training_start": configuration.training_start.isoformat(),
                    "training_end": configuration.training_end.isoformat(),
                    "test_start": configuration.test_start.isoformat(),
                    "test_end": configuration.test_end.isoformat(),
                }
            )
            mlflow.log_params(params)
            mlflow.log_dict({"features": list(feature_names)}, "config/features.json")
            mlflow.log_dict(
                {
                    "forecasting": configuration_artifact,
                    "model_parameters": dict(model_parameters),
                    "dataset_sha256": params["dataset_sha256"],
                    **dict(experiment_details or {}),
                },
                "config/experiment_config.json",
            )
        except Exception as error:
            raise MLflowTrackingError(
                f"MLflow experiment tracking failed to log run configuration: {error}"
            ) from error

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        if not self.enabled:
            return
        try:
            import mlflow

            mlflow.log_metrics(forecast_metrics(metrics))
        except ValueError:
            raise
        except Exception as error:
            raise MLflowTrackingError(
                f"MLflow experiment tracking failed to log metrics: {error}"
            ) from error

    def log_backtest_predictions(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not self.enabled:
            return
        try:
            import mlflow

            mlflow.log_dict(
                {"predictions": [dict(row) for row in rows]},
                "evaluation/forecast_vs_actual.json",
            )
        except Exception as error:
            raise MLflowTrackingError(
                f"MLflow experiment tracking failed to log evaluation artifacts: {error}"
            ) from error

    def log_rolling_window(
        self,
        *,
        configuration: ForecastRunConfiguration,
        x: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Sequence[str],
        model_parameters: Mapping[str, Any],
        requested_model_parameters: Mapping[str, Any],
        metrics: Mapping[str, float],
        predictions: Sequence[Mapping[str, Any]],
        evaluation_snapshot: Sequence[Mapping[str, Any]],
        training_snapshot: Mapping[str, Any],
        period_details: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        evaluation_sha256 = payload_fingerprint(list(evaluation_snapshot))
        predictions_sha256 = payload_fingerprint(list(predictions))
        training_refits_sha256 = payload_fingerprint(training_snapshot)
        self.log_forecast_configuration(
            configuration=configuration,
            x=x,
            y=y,
            feature_names=feature_names,
            model_parameters=model_parameters,
            experiment_details={
                "period": dict(period_details),
                "requested_model_parameters": dict(requested_model_parameters),
                "evaluation_sha256": evaluation_sha256,
                "predictions_sha256": predictions_sha256,
                "training_refits_sha256": training_refits_sha256,
            },
        )
        try:
            import mlflow

            mlflow.log_params(
                {
                    "window_index": period_details["window_index"],
                    "evaluation_sha256": evaluation_sha256,
                    "predictions_sha256": predictions_sha256,
                    "training_refits_sha256": training_refits_sha256,
                    **{
                        f"requested_xgb_{name}": value
                        for name, value in requested_model_parameters.items()
                    },
                }
            )
            mlflow.log_metrics(forecast_metrics(metrics))
            mlflow.log_dict(
                {"predictions": [dict(row) for row in predictions]},
                "evaluation/forecast_vs_actual.json",
            )
            mlflow.log_dict(
                {"rows": [dict(row) for row in evaluation_snapshot]},
                "data/evaluation_snapshot.json",
            )
            mlflow.log_dict(dict(training_snapshot), "data/training_snapshot.json")
        except Exception as error:
            raise MLflowTrackingError(
                f"MLflow experiment tracking failed to log rolling window: {error}"
            ) from error

    def log_rolling_summary(
        self,
        *,
        parameters: Mapping[str, Any],
        metrics: Mapping[str, float],
        summary: Mapping[str, Any],
        period_configuration: Mapping[str, Any],
        period_metrics_csv: str,
        predictions_csv: str,
        feature_names: Sequence[str],
        dataset_snapshot: Mapping[str, Any],
    ) -> None:
        if not self.enabled:
            return
        try:
            import mlflow

            dataset_sha256 = payload_fingerprint(dataset_snapshot)
            mlflow.log_params(
                {
                    **dict(parameters),
                    "dataset_sha256": dataset_sha256,
                    "git_commit": git_commit(),
                }
            )
            mlflow.log_metrics(forecast_metrics(metrics))
            mlflow.log_text(period_metrics_csv, "aggregate/period_metrics.csv")
            mlflow.log_text(predictions_csv, "aggregate/forecast_vs_actual.csv")
            mlflow.log_dict(dict(summary), "aggregate/summary.json")
            mlflow.log_dict(
                {
                    **dict(period_configuration),
                    "dataset_sha256": dataset_sha256,
                },
                "config/period_configuration.json",
            )
            mlflow.log_dict({"features": list(feature_names)}, "config/features.json")
            mlflow.log_dict(dict(dataset_snapshot), "data/historical_dataset.json")
        except Exception as error:
            raise MLflowTrackingError(
                f"MLflow experiment tracking failed to log rolling summary: {error}"
            ) from error
