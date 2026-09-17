from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean, pstdev

from .repository import MarketRepository

FEATURE_NAMES = (
    "price_lag_1",
    "price_lag_2",
    "price_lag_5",
    "price_lag_10",
    "price_lag_20",
    "return_1d",
    "return_5d",
    "return_20d",
    "rolling_mean_5",
    "rolling_mean_20",
    "rolling_std_5",
    "rolling_std_20",
    "price_vs_20d_average",
    "storage_total",
    "weekly_storage_change",
    "storage_vs_5yr_avg",
    "storage_vs_last_year",
    "hdd_7d",
    "hdd_14d",
    "cdd_7d",
    "cdd_14d",
    "temperature_anomaly",
    "month",
    "winter_flag",
    "summer_flag",
    "heating_season",
)


@dataclass(frozen=True)
class FeatureVector:
    as_of: datetime
    values: dict[str, float]
    latest_price: float
    latest_data_timestamp: datetime

    def ordered(self) -> list[float]:
        return [self.values[name] for name in FEATURE_NAMES]


def _safe_return(current: float, previous: float) -> float:
    return 0.0 if previous == 0 else current / previous - 1.0


class FeatureBuilder:
    def __init__(self, repository: MarketRepository) -> None:
        self.repository = repository

    def build(self, as_of: datetime) -> FeatureVector:
        prices = self.repository.prices_as_of(as_of)
        if len(prices) < 21:
            raise ValueError("at least 21 point-in-time price observations are required")
        values = [row.price for row in prices]
        latest = values[-1]
        storage = self.repository.latest_storage_as_of(as_of)
        weather = self.repository.weather_as_of(as_of, as_of.date() + timedelta(days=14))

        def degree_days(days: int, field: str) -> float:
            through = as_of.date() + timedelta(days=days)
            selected = [
                getattr(w, field) for w in weather if as_of.date() < w.observation_date <= through
            ]
            return sum(selected) / max(1, len({w.region for w in weather}))

        anomaly_values = [w.temperature_anomaly for w in weather]
        mean20 = fmean(values[-20:])
        month = float(as_of.month)
        features = {
            "price_lag_1": values[-1],
            "price_lag_2": values[-2],
            "price_lag_5": values[-5],
            "price_lag_10": values[-10],
            "price_lag_20": values[-20],
            "return_1d": _safe_return(latest, values[-2]),
            "return_5d": _safe_return(latest, values[-5]),
            "return_20d": _safe_return(latest, values[-20]),
            "rolling_mean_5": fmean(values[-5:]),
            "rolling_mean_20": mean20,
            "rolling_std_5": pstdev(values[-5:]),
            "rolling_std_20": pstdev(values[-20:]),
            "price_vs_20d_average": latest - mean20,
            "storage_total": storage.storage_bcf if storage else 0.0,
            "weekly_storage_change": storage.weekly_change_bcf if storage else 0.0,
            "storage_vs_5yr_avg": (
                storage.storage_bcf - storage.five_year_average_bcf if storage else 0.0
            ),
            "storage_vs_last_year": (
                storage.storage_bcf - storage.last_year_bcf
                if storage and storage.last_year_bcf is not None
                else 0.0
            ),
            "hdd_7d": degree_days(7, "hdd"),
            "hdd_14d": degree_days(14, "hdd"),
            "cdd_7d": degree_days(7, "cdd"),
            "cdd_14d": degree_days(14, "cdd"),
            "temperature_anomaly": fmean(anomaly_values) if anomaly_values else 0.0,
            "month": month,
            "winter_flag": float(as_of.month in {12, 1, 2}),
            "summer_flag": float(as_of.month in {6, 7, 8}),
            "heating_season": float(as_of.month in {11, 12, 1, 2, 3}),
        }
        if not all(math.isfinite(v) for v in features.values()):
            raise ValueError("feature vector contains a non-finite value")
        latest_timestamp = max(
            [p.publication_timestamp for p in prices]
            + ([storage.publication_timestamp] if storage else [])
            + [w.publication_timestamp for w in weather]
        )
        if latest_timestamp > as_of:
            raise AssertionError("point-in-time violation")
        return FeatureVector(as_of, features, latest, latest_timestamp)
