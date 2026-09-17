from __future__ import annotations

from collections.abc import Sequence

try:
    import xgboost as xgb
except ImportError as error:  # pragma: no cover - optional model
    raise RuntimeError("Install the ML extra: pip install -e '.[ml]'") from error


class XGBoostRegressor:
    """Optional production-candidate adapter matching the core regressor protocol."""

    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 3,
        learning_rate: float = 0.04,
        random_state: int = 42,
    ) -> None:
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
        )

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> XGBoostRegressor:
        self.model.fit(x, y)
        return self

    def predict(self, x: Sequence[float]) -> float:
        return float(self.model.predict([list(x)])[0])

    def contributions(self, x: Sequence[float]) -> list[float]:
        """Return exact TreeSHAP contributions via XGBoost's native predictor."""
        matrix = xgb.DMatrix([list(x)])
        values = self.model.get_booster().predict(matrix, pred_contribs=True)[0]
        return [float(value) for value in values[:-1]]
