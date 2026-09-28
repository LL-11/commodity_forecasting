from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from statistics import fmean, pstdev
from typing import Any, cast

from .backtest import interval_forecast_metrics, point_forecast_metrics
from .features import FEATURE_NAMES, FeatureBuilder
from .forecasting import DirectForecastModel, quantile
from .repository import MarketRepository
from .tracking import ForecastRunConfiguration, MLflowTracker


@dataclass(frozen=True)
class HistoricalOrigin:
    origin_price_index: int
    target_price_index: int
    origin_date: date
    origin_cutoff: datetime
    origin_publication_timestamp: datetime
    target_date: date
    target_publication_timestamp: datetime
    features: tuple[float, ...]
    actual: float
    naive_baseline: float
    moving_average_baseline: float
    origin_source: str
    target_source: str


@dataclass(frozen=True)
class HistoricalDataset:
    rows: tuple[HistoricalOrigin, ...]
    horizon: int
    evaluation_as_of: datetime
    data_kind: str
    skipped_origins: Mapping[str, int]


@dataclass(frozen=True)
class RollingWindow:
    window_index: int
    test_start_index: int
    test_end_index: int


@dataclass(frozen=True)
class RefitSnapshot:
    forecast_origin_index: int
    training_indices: tuple[int, ...]


@dataclass(frozen=True)
class RollingPrediction:
    window_index: int
    row_index: int
    origin_date: date
    target_date: date
    origin_cutoff: datetime
    target_publication_timestamp: datetime
    actual: float
    predicted: float
    naive_baseline: float
    moving_average_baseline: float
    p10: float | None
    p90: float | None
    calibration_observations: int
    calibration_max_target_publication: datetime | None


@dataclass(frozen=True)
class PeriodEvaluation:
    window: RollingWindow
    predictions: tuple[RollingPrediction, ...]
    metrics: Mapping[str, float]
    refits: tuple[RefitSnapshot, ...]
    final_training_indices: tuple[int, ...]
    train_start: date
    train_end: date


@dataclass(frozen=True)
class RollingEvaluation:
    dataset: HistoricalDataset
    periods: tuple[PeriodEvaluation, ...]
    summary: Mapping[str, Any]
    trailing_origins_skipped: int


@dataclass(frozen=True)
class TrackedRollingResult:
    evaluation: RollingEvaluation
    parent_run_id: str | None
    child_run_ids: tuple[str, ...]


def build_historical_dataset(
    repository: MarketRepository,
    horizon: int,
    evaluation_as_of: datetime | None = None,
) -> HistoricalDataset:
    """Build auditable origin/target rows using point-in-time features."""
    if horizon not in {1, 5, 20}:
        raise ValueError("horizon must be 1, 5, or 20")
    final_as_of = evaluation_as_of or repository.latest_price_publication_timestamp()
    if final_as_of is None:
        raise ValueError("rolling backtest requires price observations")
    # A historical label must use the revision that first became observable.
    # Selecting the latest revision here can make a later bulk ingestion replace
    # an earlier point-in-time-safe value and render every training label ineligible.
    prices = repository.first_published_prices_as_of(final_as_of)
    if len(prices) < 21 + horizon:
        raise ValueError(
            f"rolling backtest needs at least {21 + horizon} published price rows; "
            f"found {len(prices)}"
        )
    builder = FeatureBuilder(repository)
    rows: list[HistoricalOrigin] = []
    skipped = {
        "origin_not_published_by_cutoff": 0,
        "features_unavailable": 0,
        "target_unavailable_by_evaluation_as_of": 0,
    }
    for origin_index in range(20, len(prices) - horizon):
        origin_snapshot = prices[origin_index]
        target = prices[origin_index + horizon]
        cutoff = datetime.combine(origin_snapshot.observation_date, time.max, tzinfo=UTC)
        visible_origin = repository.prices_as_of(
            cutoff,
            start=origin_snapshot.observation_date,
            end=origin_snapshot.observation_date,
        )
        if not visible_origin:
            skipped["origin_not_published_by_cutoff"] += 1
            continue
        if target.publication_timestamp > final_as_of:
            skipped["target_unavailable_by_evaluation_as_of"] += 1
            continue
        try:
            features = tuple(builder.build(cutoff).ordered())
        except ValueError:
            skipped["features_unavailable"] += 1
            continue
        rows.append(
            HistoricalOrigin(
                origin_price_index=origin_index,
                target_price_index=origin_index + horizon,
                origin_date=origin_snapshot.observation_date,
                origin_cutoff=cutoff,
                origin_publication_timestamp=visible_origin[-1].publication_timestamp,
                target_date=target.observation_date,
                target_publication_timestamp=target.publication_timestamp,
                features=features,
                actual=target.price,
                naive_baseline=features[0],
                moving_average_baseline=features[8],
                origin_source=visible_origin[-1].source,
                target_source=target.source,
            )
        )
    if not rows:
        raise ValueError("no point-in-time historical forecast origins could be assembled")
    sources = {row.origin_source for row in rows} | {row.target_source for row in rows}
    data_kind = "synthetic" if sources == {"synthetic-demo"} else "live"
    return HistoricalDataset(tuple(rows), horizon, final_as_of, data_kind, skipped)


def eligible_training_indices(
    rows: Sequence[HistoricalOrigin], origin_index: int, train_window: int
) -> tuple[int, ...]:
    """Return the latest fixed-width labeled rows available at an origin."""
    origin = rows[origin_index]
    eligible = [
        index
        for index, candidate in enumerate(rows[:origin_index])
        if candidate.target_price_index <= origin.origin_price_index
        and candidate.target_publication_timestamp <= origin.origin_cutoff
    ]
    return tuple(eligible[-train_window:])


def plan_rolling_windows(
    dataset: HistoricalDataset,
    train_window: int,
    test_window: int,
    step: int,
    *,
    minimum_calibration_residuals: int = 10,
    minimum_periods: int = 2,
) -> tuple[tuple[RollingWindow, ...], int]:
    if train_window < 40:
        raise ValueError("train-window must be at least 40")
    if test_window < 1 or step < 1:
        raise ValueError("test-window and step must be positive")
    if step < test_window:
        raise ValueError("step must be at least test-window for disjoint v1 evaluations")
    rows = dataset.rows
    first_start: int | None = None
    for start in range(len(rows)):
        if len(eligible_training_indices(rows, start, train_window)) < train_window:
            continue
        available_calibration = 0
        for index in range(start):
            if len(eligible_training_indices(rows, index, train_window)) < train_window:
                continue
            candidate = rows[index]
            if (
                candidate.target_price_index <= rows[start].origin_price_index
                and candidate.target_publication_timestamp <= rows[start].origin_cutoff
            ):
                available_calibration += 1
        if available_calibration >= minimum_calibration_residuals:
            first_start = start
            break
    if first_start is None:
        raise ValueError(
            "not enough history for the requested training window and ten prior-only "
            "calibration residuals"
        )
    windows: list[RollingWindow] = []
    trailing = 0
    for start in range(first_start, len(rows), step):
        end = start + test_window
        if end > len(rows):
            trailing = len(rows) - start
            break
        windows.append(RollingWindow(len(windows), start, end))
    if len(windows) < minimum_periods:
        raise ValueError(
            f"rolling backtest requires at least {minimum_periods} complete periods; "
            f"found {len(windows)} with {trailing} trailing origins"
        )
    return tuple(windows), trailing


def _predict_origins(
    dataset: HistoricalDataset,
    origin_indices: Sequence[int],
    train_window: int,
    model_factory: Callable[[int], DirectForecastModel],
    retraining_frequency: int,
) -> tuple[dict[int, float], tuple[RefitSnapshot, ...]]:
    predictions: dict[int, float] = {}
    refits: list[RefitSnapshot] = []
    model: DirectForecastModel | None = None
    for offset, origin_index in enumerate(origin_indices):
        training = eligible_training_indices(dataset.rows, origin_index, train_window)
        if len(training) < train_window:
            raise ValueError(
                f"origin {dataset.rows[origin_index].origin_date} has only "
                f"{len(training)} eligible training rows; {train_window} required"
            )
        if model is None or offset % retraining_frequency == 0:
            x = [dataset.rows[index].features for index in training]
            y = [dataset.rows[index].actual for index in training]
            model = model_factory(dataset.horizon).fit(x, y)
            refits.append(RefitSnapshot(origin_index, training))
        predictions[origin_index] = max(
            0.0, model.regressor.predict(dataset.rows[origin_index].features)
        )
    return predictions, tuple(refits)


def evaluate_period(
    dataset: HistoricalDataset,
    window: RollingWindow,
    train_window: int,
    model_factory: Callable[[int], DirectForecastModel],
    *,
    retraining_frequency: int = 5,
    minimum_calibration_residuals: int = 10,
) -> PeriodEvaluation:
    calibration_origins = [
        index
        for index in range(window.test_start_index)
        if len(eligible_training_indices(dataset.rows, index, train_window)) == train_window
    ]
    calibration_predictions, _ = _predict_origins(
        dataset,
        calibration_origins,
        train_window,
        model_factory,
        retraining_frequency,
    )
    test_indices = list(range(window.test_start_index, window.test_end_index))
    test_predictions, refits = _predict_origins(
        dataset, test_indices, train_window, model_factory, retraining_frequency
    )
    prior_predictions = dict(calibration_predictions)
    predictions: list[RollingPrediction] = []
    for row_index in test_indices:
        row = dataset.rows[row_index]
        eligible_residual_rows = [
            index
            for index in sorted(prior_predictions)
            if index < row_index
            and dataset.rows[index].target_price_index <= row.origin_price_index
            and dataset.rows[index].target_publication_timestamp <= row.origin_cutoff
        ]
        residuals = [
            dataset.rows[index].actual - prior_predictions[index]
            for index in eligible_residual_rows
        ]
        point = test_predictions[row_index]
        p10: float | None = None
        p90: float | None = None
        maximum_publication: datetime | None = None
        if len(residuals) >= minimum_calibration_residuals:
            p10 = max(0.0, point + quantile(residuals, 0.10))
            p90 = max(p10, point + quantile(residuals, 0.90))
            maximum_publication = max(
                dataset.rows[index].target_publication_timestamp for index in eligible_residual_rows
            )
        predictions.append(
            RollingPrediction(
                window_index=window.window_index,
                row_index=row_index,
                origin_date=row.origin_date,
                target_date=row.target_date,
                origin_cutoff=row.origin_cutoff,
                target_publication_timestamp=row.target_publication_timestamp,
                actual=row.actual,
                predicted=point,
                naive_baseline=row.naive_baseline,
                moving_average_baseline=row.moving_average_baseline,
                p10=p10,
                p90=p90,
                calibration_observations=len(residuals),
                calibration_max_target_publication=maximum_publication,
            )
        )
        prior_predictions[row_index] = point
    point_metrics = point_forecast_metrics(
        [row.actual for row in predictions],
        [row.predicted for row in predictions],
        [row.naive_baseline for row in predictions],
        [row.moving_average_baseline for row in predictions],
    )
    interval_rows = [row for row in predictions if row.p10 is not None and row.p90 is not None]
    interval_metrics = interval_forecast_metrics(
        [row.actual for row in interval_rows],
        [cast(float, row.p10) for row in interval_rows],
        [cast(float, row.p90) for row in interval_rows],
    )
    metrics = {
        "test_mae": point_metrics["mae"],
        "test_rmse": point_metrics["rmse"],
        "test_mape": point_metrics["mape"],
        "test_smape": point_metrics["smape"],
        "naive_mae": point_metrics["baseline_mae"],
        "moving_average_mae": point_metrics["moving_average_baseline_mae"],
        "skill_vs_naive": point_metrics["skill_vs_naive"],
        "skill_vs_moving_average": point_metrics["skill_vs_moving_average"],
        "directional_accuracy": point_metrics["directional_accuracy"],
        "test_observations": point_metrics["observations"],
        **interval_metrics,
    }
    training_union = sorted({index for refit in refits for index in refit.training_indices})
    final_training = refits[-1].training_indices
    return PeriodEvaluation(
        window=window,
        predictions=tuple(predictions),
        metrics=metrics,
        refits=refits,
        final_training_indices=final_training,
        train_start=dataset.rows[training_union[0]].origin_date,
        train_end=dataset.rows[training_union[-1]].origin_date,
    )


def aggregate_periods(
    dataset: HistoricalDataset,
    periods: Sequence[PeriodEvaluation],
    trailing_origins_skipped: int,
) -> dict[str, Any]:
    pooled = [prediction for period in periods for prediction in period.predictions]
    point = point_forecast_metrics(
        [row.actual for row in pooled],
        [row.predicted for row in pooled],
        [row.naive_baseline for row in pooled],
        [row.moving_average_baseline for row in pooled],
    )
    interval_rows = [row for row in pooled if row.p10 is not None and row.p90 is not None]
    intervals = interval_forecast_metrics(
        [row.actual for row in interval_rows],
        [cast(float, row.p10) for row in interval_rows],
        [cast(float, row.p90) for row in interval_rows],
    )
    pooled_metrics = {
        "test_mae": point["mae"],
        "test_rmse": point["rmse"],
        "test_mape": point["mape"],
        "test_smape": point["smape"],
        "naive_mae": point["baseline_mae"],
        "moving_average_mae": point["moving_average_baseline_mae"],
        "skill_vs_naive": point["skill_vs_naive"],
        "skill_vs_moving_average": point["skill_vs_moving_average"],
        "directional_accuracy": point["directional_accuracy"],
        "test_observations": point["observations"],
        **intervals,
    }
    statistic_names = [
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_smape",
        "naive_mae",
        "moving_average_mae",
        "skill_vs_naive",
        "skill_vs_moving_average",
        "directional_accuracy",
        "p10_pinball_loss",
        "p90_pinball_loss",
        "prediction_interval_coverage",
        "prediction_interval_width",
    ]
    lower_is_worse = {
        "skill_vs_naive",
        "skill_vs_moving_average",
        "directional_accuracy",
        "prediction_interval_coverage",
    }
    period_statistics: dict[str, dict[str, float | int]] = {}
    for name in statistic_names:
        values = [
            (period.window.window_index, float(period.metrics[name]))
            for period in periods
            if name in period.metrics
        ]
        if not values:
            continue
        scores = [value for _, value in values]
        worst_stat = (
            min(values, key=lambda pair: pair[1])
            if name in lower_is_worse
            else max(values, key=lambda pair: pair[1])
        )
        period_statistics[name] = {
            "mean": fmean(scores),
            "std": pstdev(scores),
            "min": min(scores),
            "max": max(scores),
            "worst_window_index": worst_stat[0],
        }
    best_period = min(periods, key=lambda period: period.metrics["test_mae"])
    worst_period = max(periods, key=lambda period: period.metrics["test_mae"])
    return {
        "horizon": dataset.horizon,
        "evaluation_as_of": dataset.evaluation_as_of.isoformat(),
        "data_kind": dataset.data_kind,
        "period_count": len(periods),
        "prediction_count": len(pooled),
        "interval_observation_count": len(interval_rows),
        "trailing_origins_skipped": trailing_origins_skipped,
        "skipped_origin_reasons": dict(dataset.skipped_origins),
        "pooled_metrics": pooled_metrics,
        "period_statistics": period_statistics,
        "best_period_by_mae": best_period.window.window_index,
        "worst_period_by_mae": worst_period.window.window_index,
    }


def run_rolling_evaluation(
    dataset: HistoricalDataset,
    train_window: int,
    test_window: int,
    step: int,
    model_factory: Callable[[int], DirectForecastModel],
    *,
    retraining_frequency: int = 5,
) -> RollingEvaluation:
    windows, trailing = plan_rolling_windows(dataset, train_window, test_window, step)
    periods = tuple(
        evaluate_period(
            dataset,
            window,
            train_window,
            model_factory,
            retraining_frequency=retraining_frequency,
        )
        for window in windows
    )
    return RollingEvaluation(
        dataset,
        periods,
        aggregate_periods(dataset, periods, trailing),
        trailing,
    )


def prediction_artifact(prediction: RollingPrediction) -> dict[str, Any]:
    return {
        "window_index": prediction.window_index,
        "origin_date": prediction.origin_date.isoformat(),
        "target_date": prediction.target_date.isoformat(),
        "origin_cutoff": prediction.origin_cutoff.isoformat(),
        "target_publication_timestamp": prediction.target_publication_timestamp.isoformat(),
        "actual": prediction.actual,
        "predicted": prediction.predicted,
        "naive_baseline": prediction.naive_baseline,
        "moving_average_baseline": prediction.moving_average_baseline,
        "p10": prediction.p10,
        "p90": prediction.p90,
        "calibration_observations": prediction.calibration_observations,
        "calibration_max_target_publication": (
            prediction.calibration_max_target_publication.isoformat()
            if prediction.calibration_max_target_publication
            else None
        ),
    }


def source_row_artifact(row: HistoricalOrigin) -> dict[str, Any]:
    return {
        "origin_price_index": row.origin_price_index,
        "target_price_index": row.target_price_index,
        "origin_date": row.origin_date.isoformat(),
        "origin_cutoff": row.origin_cutoff.isoformat(),
        "origin_publication_timestamp": row.origin_publication_timestamp.isoformat(),
        "target_date": row.target_date.isoformat(),
        "target_publication_timestamp": row.target_publication_timestamp.isoformat(),
        "features": dict(zip(FEATURE_NAMES, row.features, strict=True)),
        "actual": row.actual,
        "naive_baseline": row.naive_baseline,
        "moving_average_baseline": row.moving_average_baseline,
        "origin_source": row.origin_source,
        "target_source": row.target_source,
    }


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


class RollingBacktestService:
    """Run pure rolling evaluation first, then record parent/child MLflow runs."""

    def __init__(
        self,
        repository: MarketRepository,
        tracker: MLflowTracker,
        xgboost_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.repository = repository
        self.tracker = tracker
        self.xgboost_params = dict(xgboost_params or {})

    def _model(self, horizon: int) -> DirectForecastModel:
        from .xgboost_model import XGBoostRegressor

        return DirectForecastModel(horizon, regressor=XGBoostRegressor(**self.xgboost_params))

    def _actual_model_parameters(self) -> dict[str, Any]:
        from .xgboost_model import XGBoostRegressor

        return dict(XGBoostRegressor(**self.xgboost_params).model.get_params())

    def run(
        self,
        *,
        horizon: int,
        train_window: int,
        test_window: int,
        step: int,
        run_name: str | None,
    ) -> TrackedRollingResult:
        dataset = build_historical_dataset(self.repository, horizon)
        evaluation = run_rolling_evaluation(
            dataset,
            train_window,
            test_window,
            step,
            self._model,
        )
        if not self.tracker.enabled:
            return TrackedRollingResult(
                evaluation=evaluation,
                parent_run_id=None,
                child_run_ids=(),
            )
        group = run_name or f"xgb-{horizon}d-rolling"
        actual_model_parameters = self._actual_model_parameters()
        parent_tags = {
            "run_type": "rolling_summary",
            "data_kind": dataset.data_kind,
            "comparison_group": group,
        }
        child_run_ids: list[str] = []
        period_rows: list[dict[str, Any]] = []
        aggregate_prediction_rows: list[dict[str, Any]] = []
        with self.tracker.start_run(
            run_name=run_name,
            model_name="rolling_summary",
            tags=parent_tags,
        ) as parent_run_id:
            if parent_run_id is None:
                raise RuntimeError("rolling backtest requires MLflow tracking")
            for period in evaluation.periods:
                window_index = period.window.window_index
                predictions = [prediction_artifact(row) for row in period.predictions]
                evaluation_snapshot = [
                    {
                        "row_index": prediction.row_index,
                        **source_row_artifact(dataset.rows[prediction.row_index]),
                    }
                    for prediction in period.predictions
                ]
                training_indices = sorted(
                    {index for refit in period.refits for index in refit.training_indices}
                )
                training_snapshot = {
                    "refits": [
                        {
                            "forecast_origin": dataset.rows[
                                refit.forecast_origin_index
                            ].origin_date.isoformat(),
                            "training_row_indices": list(refit.training_indices),
                        }
                        for refit in period.refits
                    ],
                    "rows": [
                        {"row_index": index, **source_row_artifact(dataset.rows[index])}
                        for index in training_indices
                    ],
                    "final_training_indices": list(period.final_training_indices),
                }
                final_rows = [dataset.rows[index] for index in period.final_training_indices]
                x = [row.features for row in final_rows]
                y = [row.actual for row in final_rows]
                configuration = ForecastRunConfiguration(
                    target="henry_hub_spot_price_usd_per_mmbtu",
                    forecast_horizon=horizon,
                    as_of=dataset.evaluation_as_of,
                    training_start=period.train_start,
                    training_end=period.train_end,
                    test_start=period.predictions[0].origin_date,
                    test_end=period.predictions[-1].origin_date,
                    model_name="xgboost",
                )
                period_details = {
                    "window_index": window_index,
                    "origin_cutoff_start": period.predictions[0].origin_cutoff.isoformat(),
                    "origin_cutoff_end": period.predictions[-1].origin_cutoff.isoformat(),
                    "target_date_end": period.predictions[-1].target_date.isoformat(),
                    "train_window": train_window,
                    "test_window": test_window,
                    "step": step,
                    "retraining_frequency": 5,
                    "model_artifact_role": "final_period_refit",
                }
                child_tags = {
                    "run_type": "rolling_window",
                    "window_index": str(window_index),
                    "horizon": str(horizon),
                    "parent_run_id": parent_run_id,
                    "comparison_group": group,
                }
                child_name = f"{group}-window-{window_index:02d}"
                with self.tracker.start_run(
                    run_name=child_name,
                    model_name="xgboost",
                    nested=True,
                    tags=child_tags,
                ) as child_run_id:
                    if child_run_id is None:
                        raise RuntimeError("rolling child run was not created")
                    # Refit the final period model once inside the child run so
                    # XGBoost autologging associates exactly one model with it.
                    self._model(horizon).fit(x, y)
                    self.tracker.log_rolling_window(
                        configuration=configuration,
                        x=x,
                        y=y,
                        feature_names=FEATURE_NAMES,
                        model_parameters=actual_model_parameters,
                        requested_model_parameters=self.xgboost_params,
                        metrics=period.metrics,
                        predictions=predictions,
                        evaluation_snapshot=evaluation_snapshot,
                        training_snapshot=training_snapshot,
                        period_details=period_details,
                    )
                child_run_ids.append(child_run_id)
                period_rows.append(
                    {
                        "window_index": window_index,
                        "train_start": period.train_start.isoformat(),
                        "train_end": period.train_end.isoformat(),
                        "test_start": period.predictions[0].origin_date.isoformat(),
                        "test_end": period.predictions[-1].origin_date.isoformat(),
                        "training_rows": len(period.final_training_indices),
                        "test_observations": int(period.metrics["test_observations"]),
                        "interval_observations": int(period.metrics["interval_observations"]),
                        "test_mae": period.metrics["test_mae"],
                        "test_rmse": period.metrics["test_rmse"],
                        "test_mape": period.metrics["test_mape"],
                        "test_smape": period.metrics["test_smape"],
                        "naive_mae": period.metrics["naive_mae"],
                        "moving_average_mae": period.metrics["moving_average_mae"],
                        "skill_vs_naive": period.metrics["skill_vs_naive"],
                        "skill_vs_moving_average": period.metrics["skill_vs_moving_average"],
                        "directional_accuracy": period.metrics["directional_accuracy"],
                        "p10_pinball_loss": period.metrics.get("p10_pinball_loss"),
                        "p90_pinball_loss": period.metrics.get("p90_pinball_loss"),
                        "interval_coverage": period.metrics.get("prediction_interval_coverage"),
                        "interval_width": period.metrics.get("prediction_interval_width"),
                        "mlflow_run_id": child_run_id,
                    }
                )
                aggregate_prediction_rows.extend(
                    {
                        **row,
                        "mlflow_run_id": child_run_id,
                    }
                    for row in predictions
                )
            summary = {
                **dict(evaluation.summary),
                "parent_run_id": parent_run_id,
                "child_run_ids": child_run_ids,
                "periods": period_rows,
            }
            tracked_parameter_names = (
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "gamma",
                "reg_alpha",
                "reg_lambda",
                "objective",
                "random_state",
            )
            parent_parameters = {
                "target": "henry_hub_spot_price_usd_per_mmbtu",
                "forecast_horizon": horizon,
                "train_window": train_window,
                "test_window": test_window,
                "step": step,
                "feature_count": len(FEATURE_NAMES),
                "period_count": len(evaluation.periods),
                "data_kind": dataset.data_kind,
                **{
                    f"xgb_{name}": actual_model_parameters[name] for name in tracked_parameter_names
                },
            }
            period_configuration = {
                "horizon": horizon,
                "train_window": train_window,
                "test_window": test_window,
                "step": step,
                "retraining_frequency": 5,
                "minimum_calibration_residuals": 10,
                "evaluation_as_of": dataset.evaluation_as_of.isoformat(),
                "data_kind": dataset.data_kind,
                "requested_xgboost_parameters": self.xgboost_params,
                "actual_xgboost_parameters": actual_model_parameters,
                "features": list(FEATURE_NAMES),
            }
            dataset_snapshot = {
                "horizon": dataset.horizon,
                "evaluation_as_of": dataset.evaluation_as_of.isoformat(),
                "data_kind": dataset.data_kind,
                "skipped_origins": dict(dataset.skipped_origins),
                "rows": [
                    {"row_index": index, **source_row_artifact(row)}
                    for index, row in enumerate(dataset.rows)
                ],
            }
            self.tracker.log_rolling_summary(
                parameters=parent_parameters,
                metrics=evaluation.summary["pooled_metrics"],
                summary=summary,
                period_configuration=period_configuration,
                period_metrics_csv=_csv_text(period_rows),
                predictions_csv=_csv_text(aggregate_prediction_rows),
                feature_names=FEATURE_NAMES,
                dataset_snapshot=dataset_snapshot,
            )
        return TrackedRollingResult(
            evaluation=evaluation,
            parent_run_id=parent_run_id,
            child_run_ids=tuple(child_run_ids),
        )
