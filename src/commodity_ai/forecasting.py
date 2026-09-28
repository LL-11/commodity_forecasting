from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Protocol


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return fmean(abs(a - p) for a, p in zip(actual, predicted, strict=True))


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return math.sqrt(fmean((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)))


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    terms = [2 * abs(a - p) / (abs(a) + abs(p)) for a, p in zip(actual, predicted) if a or p]
    return fmean(terms) if terms else 0.0


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error as a fraction, excluding zero actuals."""
    terms = [abs(a - p) / abs(a) for a, p in zip(actual, predicted, strict=True) if a != 0]
    return fmean(terms) if terms else 0.0


class Regressor(Protocol):
    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> Regressor: ...
    def predict(self, x: Sequence[float]) -> float: ...
    def contributions(self, x: Sequence[float]) -> Sequence[float]: ...


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a dense linear system with partial-pivot Gaussian elimination."""
    n = len(vector)
    augmented = [matrix[i][:] + [vector[i]] for i in range(n)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if abs(augmented[column][column]) < 1e-12:
            augmented[column][column] = 1e-12
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][j] - factor * augmented[column][j] for j in range(n + 1)
            ]
    return [augmented[i][-1] for i in range(n)]


class RidgeRegressor:
    """Small dependency-free ridge model used as the initial direct forecaster."""

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.means: list[float] = []
        self.scales: list[float] = []
        self.coefficients: list[float] = []
        self.intercept = 0.0

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> RidgeRegressor:
        if not x or len(x) != len(y):
            raise ValueError("training features and targets must be non-empty and aligned")
        width = len(x[0])
        self.means = [fmean(row[j] for row in x) for j in range(width)]
        self.scales = []
        for j in range(width):
            variance = fmean((row[j] - self.means[j]) ** 2 for row in x)
            self.scales.append(math.sqrt(variance) or 1.0)
        normalized = [
            [(row[j] - self.means[j]) / self.scales[j] for j in range(width)] for row in x
        ]
        self.intercept = fmean(y)
        centered_y = [value - self.intercept for value in y]
        gram = [[0.0 for _ in range(width)] for _ in range(width)]
        rhs = [0.0 for _ in range(width)]
        for row, target in zip(normalized, centered_y, strict=True):
            for j in range(width):
                rhs[j] += row[j] * target
                for k in range(width):
                    gram[j][k] += row[j] * row[k]
        for j in range(width):
            gram[j][j] += self.alpha
        self.coefficients = _solve(gram, rhs)
        return self

    def predict(self, x: Sequence[float]) -> float:
        return self.intercept + sum(self.contributions(x))

    def contributions(self, x: Sequence[float]) -> list[float]:
        return [
            coefficient * (value - mean) / scale
            for coefficient, value, mean, scale in zip(
                self.coefficients, x, self.means, self.scales, strict=True
            )
        ]


@dataclass(frozen=True)
class CalibratedPrediction:
    point: float
    p10: float
    p50: float
    p90: float
    residuals: tuple[float, ...]


class DirectForecastModel:
    def __init__(
        self, horizon: int, alpha: float = 10.0, regressor: Regressor | None = None
    ) -> None:
        self.horizon = horizon
        self.regressor = regressor or RidgeRegressor(alpha)
        self.residuals: list[float] = []

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> DirectForecastModel:
        self.regressor.fit(x, y)
        # This initial calibration is replaced by out-of-sample residuals by the
        # forecasting service before a production forecast is emitted.
        self.residuals = [
            target - self.regressor.predict(row) for row, target in zip(x, y, strict=True)
        ]
        return self

    def calibrate(self, out_of_sample_residuals: Sequence[float]) -> DirectForecastModel:
        if len(out_of_sample_residuals) < 10:
            raise ValueError("at least 10 out-of-sample residuals are required for calibration")
        self.residuals = list(out_of_sample_residuals)
        return self

    def predict(self, x: Sequence[float]) -> CalibratedPrediction:
        point = max(0.0, self.regressor.predict(x))
        low = max(0.0, point + quantile(self.residuals, 0.10))
        median = max(0.0, point + quantile(self.residuals, 0.50))
        high = max(median, point + quantile(self.residuals, 0.90))
        return CalibratedPrediction(point, low, median, high, tuple(self.residuals))


def persistence(latest_price: float) -> float:
    return latest_price


def moving_average(prices: Sequence[float], window: int = 5) -> float:
    if len(prices) < window:
        raise ValueError(f"at least {window} prices are required")
    return fmean(prices[-window:])


def pinball_loss(actual: Sequence[float], predicted: Sequence[float], level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("quantile level must be between zero and one")
    losses = []
    for observed, estimate in zip(actual, predicted, strict=True):
        error = observed - estimate
        losses.append(max(level * error, (level - 1) * error))
    return fmean(losses)
