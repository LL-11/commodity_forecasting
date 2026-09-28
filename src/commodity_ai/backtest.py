from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from .forecasting import DirectForecastModel, mae, mape, pinball_loss, quantile, rmse, smape


@dataclass(frozen=True)
class BacktestPrediction:
    as_of: date
    actual: float
    predicted: float
    baseline: float
    moving_average_baseline: float
    p10: float
    p90: float


@dataclass(frozen=True)
class BacktestResult:
    predictions: tuple[BacktestPrediction, ...]
    metrics: dict[str, float]
    calibration_residuals: tuple[float, ...]


def point_forecast_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    baselines: Sequence[float],
    moving_average_baselines: Sequence[float],
) -> dict[str, float]:
    """Calculate shared point-forecast and baseline metrics."""
    if not (len(actual) == len(predicted) == len(baselines) == len(moving_average_baselines)):
        raise ValueError("metric inputs must have equal lengths")
    if not actual:
        raise ValueError("at least one prediction is required")
    model_mae = mae(actual, predicted)
    baseline_mae = mae(actual, baselines)
    moving_average_mae = mae(actual, moving_average_baselines)
    directional_accuracy = sum(
        ((estimate - base) * (observed - base)) > 0
        for observed, estimate, base in zip(actual, predicted, baselines, strict=True)
    ) / len(actual)
    return {
        "mae": model_mae,
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
        "baseline_mae": baseline_mae,
        "moving_average_baseline_mae": moving_average_mae,
        "skill_vs_naive": 1 - model_mae / baseline_mae if baseline_mae else 0.0,
        "skill_vs_moving_average": (
            1 - model_mae / moving_average_mae if moving_average_mae else 0.0
        ),
        "directional_accuracy": directional_accuracy,
        "observations": float(len(actual)),
    }


def interval_forecast_metrics(
    actual: Sequence[float], p10: Sequence[float], p90: Sequence[float]
) -> dict[str, float]:
    """Calculate interval metrics over predictions with eligible bounds."""
    if not (len(actual) == len(p10) == len(p90)):
        raise ValueError("interval metric inputs must have equal lengths")
    if not actual:
        return {"interval_observations": 0.0}
    coverage = sum(
        low <= observed <= high for low, observed, high in zip(p10, actual, p90, strict=True)
    ) / len(actual)
    width = sum(high - low for low, high in zip(p10, p90, strict=True)) / len(actual)
    return {
        "p10_pinball_loss": pinball_loss(actual, p10, 0.10),
        "p90_pinball_loss": pinball_loss(actual, p90, 0.90),
        "prediction_interval_coverage": coverage,
        "prediction_interval_width": width,
        "interval_observations": float(len(actual)),
    }


def walk_forward_backtest(
    features: Sequence[Sequence[float]],
    targets: Sequence[float],
    baselines: Sequence[float],
    dates: Sequence[date],
    forecast_horizon: int,
    retraining_frequency: int = 5,
    minimum_training_rows: int = 40,
    model_factory: Callable[[int], DirectForecastModel] = DirectForecastModel,
    moving_average_baselines: Sequence[float] | None = None,
) -> BacktestResult:
    """Expanding-window backtest that accounts for target publication lag.

    At origin ``i``, only examples whose targets are at least ``forecast_horizon``
    rows behind that origin are eligible for training.
    """
    size = len(features)
    if not (size == len(targets) == len(baselines) == len(dates)):
        raise ValueError("backtest inputs must have equal lengths")
    if moving_average_baselines is not None and len(moving_average_baselines) != size:
        raise ValueError("moving-average baseline must align with backtest inputs")
    average_baselines = moving_average_baselines or baselines
    if forecast_horizon < 1 or retraining_frequency < 1:
        raise ValueError("horizon and retraining frequency must be positive")
    predictions: list[BacktestPrediction] = []
    model: DirectForecastModel | None = None
    last_train_end = -1
    for origin in range(size):
        train_end = origin - forecast_horizon + 1
        if train_end < minimum_training_rows:
            continue
        if model is None or train_end - last_train_end >= retraining_frequency:
            model = model_factory(forecast_horizon).fit(features[:train_end], targets[:train_end])
            last_train_end = train_end
        result = model.predict(features[origin])
        predictions.append(
            BacktestPrediction(
                dates[origin],
                targets[origin],
                result.point,
                baselines[origin],
                average_baselines[origin],
                result.p10,
                result.p90,
            )
        )
    if not predictions:
        raise ValueError("not enough rows for the requested backtest")
    actual = [row.actual for row in predictions]
    predicted = [row.predicted for row in predictions]
    baseline = [row.baseline for row in predictions]
    moving_average_baseline = [row.moving_average_baseline for row in predictions]
    residuals = [a - p for a, p in zip(actual, predicted, strict=True)]
    residual_p10 = quantile(residuals, 0.10)
    residual_p90 = quantile(residuals, 0.90)
    calibrated_p10 = [max(0.0, value + residual_p10) for value in predicted]
    calibrated_p90 = [max(0.0, value + residual_p90) for value in predicted]
    metrics = point_forecast_metrics(actual, predicted, baseline, moving_average_baseline)
    interval_metrics = interval_forecast_metrics(actual, calibrated_p10, calibrated_p90)
    return BacktestResult(
        tuple(predictions),
        {
            **metrics,
            **interval_metrics,
        },
        tuple(residuals),
    )
