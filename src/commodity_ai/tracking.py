from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import datetime
from typing import Any


def dataset_fingerprint(x: Sequence[Sequence[float]], y: Sequence[float]) -> str:
    payload = json.dumps({"x": x, "y": y}, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


class MLflowTracker:
    def __init__(self, enabled: bool | None = None) -> None:
        configured = os.getenv("COMMODITY_AI_MLFLOW_ENABLED", "false").lower()
        self.enabled = enabled if enabled is not None else configured in {"1", "true", "yes"}

    def log_forecast_run(
        self,
        *,
        model_name: str,
        horizon: int,
        as_of: datetime,
        metrics: dict[str, float],
        x: Sequence[Sequence[float]],
        y: Sequence[float],
        feature_names: Sequence[str],
        model: Any,
    ) -> str | None:
        if not self.enabled:
            return None
        import mlflow

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///data/mlflow.db"))
        mlflow.set_experiment("henry-hub-forecasting")
        with mlflow.start_run(run_name=f"henry-hub-{model_name}-{horizon}d") as run:
            mlflow.log_params(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "as_of": as_of.isoformat(),
                    "training_rows": len(x),
                    "dataset_sha256": dataset_fingerprint(x, y),
                    "git_commit": git_commit(),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"features": list(feature_names)}, "feature_manifest.json")
            regressor = getattr(model, "regressor", None)
            booster_model = getattr(regressor, "model", None)
            if booster_model is not None and hasattr(booster_model, "get_booster"):
                raw = booster_model.get_booster().save_raw(raw_format="json")
                mlflow.log_text(bytes(raw).decode("utf-8"), "model/model.json")
            return run.info.run_id
