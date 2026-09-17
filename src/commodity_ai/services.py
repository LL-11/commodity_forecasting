from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime, time, timedelta

from .backtest import walk_forward_backtest
from .domain import Driver, ForecastRecord, utc_now
from .features import FEATURE_NAMES, FeatureBuilder
from .forecasting import DirectForecastModel, mae, persistence, rmse, smape
from .repository import MarketRepository
from .tracking import MLflowTracker


class ForecastService:
    def __init__(
        self,
        repository: MarketRepository,
        model_backend: str | None = None,
        tracker: MLflowTracker | None = None,
    ) -> None:
        self.repository = repository
        self.builder = FeatureBuilder(repository)
        self.model_backend: str = (
            model_backend or os.getenv("COMMODITY_AI_MODEL_BACKEND") or "xgboost"
        )
        if self.model_backend not in {"xgboost", "ridge"}:
            raise ValueError("model backend must be 'xgboost' or 'ridge'")
        self.model_version = f"{self.model_backend}-direct-0.2"
        self.tracker = tracker or MLflowTracker()

    def _model(self, horizon: int) -> DirectForecastModel:
        if self.model_backend == "ridge":
            return DirectForecastModel(horizon)
        from .xgboost_model import XGBoostRegressor

        return DirectForecastModel(horizon, regressor=XGBoostRegressor())

    def _training_data(
        self, as_of: datetime, horizon: int
    ) -> tuple[list[list[float]], list[float], list[date]]:
        prices = self.repository.prices_as_of(as_of)
        x: list[list[float]] = []
        y: list[float] = []
        dates: list[date] = []
        for index in range(20, len(prices) - horizon):
            cutoff = datetime.combine(prices[index].observation_date, time.max, tzinfo=UTC)
            cutoff = min(cutoff, as_of)
            try:
                row = self.builder.build(cutoff)
            except ValueError:
                continue
            x.append(row.ordered())
            y.append(prices[index + horizon].price)
            dates.append(prices[index].observation_date)
        if len(x) < 20:
            raise ValueError(
                f"horizon {horizon} needs at least 20 training examples; found {len(x)}"
            )
        return x, y, dates

    def run_forecast(self, as_of: datetime, horizons: list[int]) -> list[ForecastRecord]:
        unsupported = set(horizons) - {1, 5, 20}
        if unsupported:
            raise ValueError(f"unsupported horizons: {sorted(unsupported)}")
        current = self.builder.build(as_of)
        results: list[ForecastRecord] = []
        for horizon in horizons:
            x, y, dates = self._training_data(as_of, horizon)
            historical_result = walk_forward_backtest(
                x,
                y,
                [row[0] for row in x],
                dates,
                horizon,
                retraining_frequency=5,
                minimum_training_rows=min(40, len(x) // 2),
                model_factory=self._model,
                moving_average_baselines=[row[8] for row in x],
            )
            model = self._model(horizon).fit(x, y)
            model.calibrate(historical_result.calibration_residuals)
            prediction = model.predict(current.ordered())
            fitted = [model.regressor.predict(row) for row in x]
            naive = [row[0] for row in x]
            model_mae = mae(y, fitted)
            naive_mae = mae(y, naive)
            historical = historical_result.metrics
            contributions = model.regressor.contributions(current.ordered())
            ranked = sorted(
                zip(FEATURE_NAMES, contributions), key=lambda pair: abs(pair[1]), reverse=True
            )
            drivers = tuple(
                Driver(name, effect, "positive" if effect >= 0 else "negative")
                for name, effect in ranked[:8]
            )
            metrics = {
                "training_rows": float(len(x)),
                "walk_forward_mae": historical["mae"],
                "walk_forward_rmse": historical["rmse"],
                "walk_forward_smape": historical["smape"],
                "walk_forward_naive_mae": historical["baseline_mae"],
                "walk_forward_moving_average_mae": historical["moving_average_baseline_mae"],
                "walk_forward_skill_vs_naive": historical["skill_vs_naive"],
                "walk_forward_skill_vs_moving_average": historical["skill_vs_moving_average"],
                "walk_forward_interval_coverage": historical["prediction_interval_coverage"],
                "walk_forward_interval_width": historical["prediction_interval_width"],
                "walk_forward_directional_accuracy": historical["directional_accuracy"],
                "walk_forward_p10_pinball_loss": historical["p10_pinball_loss"],
                "walk_forward_p90_pinball_loss": historical["p90_pinball_loss"],
                "walk_forward_observations": historical["observations"],
                "fit_mae": model_mae,
                "fit_rmse": rmse(y, fitted),
                "fit_smape": smape(y, fitted),
                "fit_naive_mae": naive_mae,
            }
            run_id = self.tracker.log_forecast_run(
                model_name=self.model_backend,
                horizon=horizon,
                as_of=as_of,
                metrics=metrics,
                x=x,
                y=y,
                feature_names=FEATURE_NAMES,
                model=model,
            )
            if run_id:
                metrics["mlflow_run_recorded"] = 1.0
            record = ForecastRecord(
                forecast_id=str(uuid.uuid4()),
                as_of_timestamp=as_of,
                latest_data_timestamp=current.latest_data_timestamp,
                model_version=self.model_version,
                horizon=horizon,
                point_forecast=prediction.point,
                p10=prediction.p10,
                p50=prediction.p50,
                p90=prediction.p90,
                baseline_forecast=persistence(current.latest_price),
                created_timestamp=utc_now(),
                drivers=drivers,
                metrics=metrics,
            )
            self.repository.save_forecast(record)
            results.append(record)
        return results

    def evaluate_forecast(self, forecast_id: str, evaluation_as_of: datetime) -> dict[str, object]:
        forecast = self.repository.get_forecast(forecast_id)
        if forecast is None:
            raise ValueError("forecast not found")
        prices = self.repository.prices_as_of(evaluation_as_of)
        origins = [
            index
            for index, row in enumerate(prices)
            if row.observation_date <= forecast.as_of_timestamp.date()
        ]
        if not origins:
            raise ValueError("forecast origin price is unavailable")
        origin_index = origins[-1]
        target_index = origin_index + forecast.horizon
        if target_index >= len(prices):
            raise ValueError("actual price is not available yet")
        origin = prices[origin_index].price
        actual = prices[target_index].price
        error = forecast.point_forecast - actual
        direction_correct = ((forecast.point_forecast - origin) * (actual - origin)) > 0
        self.repository.save_forecast_actual(
            forecast_id, actual, error, direction_correct, evaluation_as_of
        )
        return {
            "forecast_id": forecast_id,
            "horizon": forecast.horizon,
            "actual_price": actual,
            "error": error,
            "absolute_error": abs(error),
            "squared_error": error**2,
            "direction_correct": direction_correct,
            "evaluated_timestamp": evaluation_as_of.isoformat(),
        }


class MarketService:
    def __init__(self, repository: MarketRepository) -> None:
        self.repository = repository

    def snapshot(self, as_of: datetime) -> dict[str, object]:
        prices = self.repository.prices_as_of(as_of)
        if not prices:
            raise ValueError("no price data available")
        storage = self.repository.latest_storage_as_of(as_of)
        return {
            "as_of": as_of.isoformat(),
            "latest_price": prices[-1].price,
            "price_date": prices[-1].observation_date.isoformat(),
            "storage_bcf": storage.storage_bcf if storage else None,
            "storage_vs_5yr_avg_bcf": (
                storage.storage_bcf - storage.five_year_average_bcf if storage else None
            ),
            "storage_publication_timestamp": (
                storage.publication_timestamp.isoformat() if storage else None
            ),
            "data_freshness": {
                "price": prices[-1].publication_timestamp.isoformat(),
                "storage": storage.publication_timestamp.isoformat() if storage else None,
            },
        }

    def weather_signal(self, as_of: datetime, horizon: int) -> dict[str, object]:
        if horizon not in {7, 14}:
            raise ValueError("weather horizon must be 7 or 14 days")
        rows = self.repository.weather_as_of(as_of, as_of.date() + timedelta(days=horizon))
        future = [row for row in rows if as_of.date() < row.observation_date]
        regions = len({row.region for row in future}) or 1
        latest_publication = max((row.publication_timestamp for row in future), default=None)
        return {
            "as_of": as_of.isoformat(),
            "horizon_days": horizon,
            "hdd": sum(row.hdd for row in future) / regions,
            "cdd": sum(row.cdd for row in future) / regions,
            "mean_temperature_anomaly": (
                sum(row.temperature_anomaly for row in future) / len(future) if future else None
            ),
            "regions": sorted({row.region for row in future}),
            "latest_publication_timestamp": (
                latest_publication.isoformat() if latest_publication else None
            ),
        }
