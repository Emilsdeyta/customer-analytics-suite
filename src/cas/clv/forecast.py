"""ML-based CLV baseline: RFM features -> XGBoost regression."""
from __future__ import annotations

from xgboost import XGBRegressor


def train_baseline(X, y, random_seed: int = 42) -> XGBRegressor:
    """Fit on the given X/y directly — caller controls the train/test split so that
    comparisons against other models (e.g. BTYD) are evaluated on the exact same holdout."""
    model = XGBRegressor(random_state=random_seed)
    model.fit(X, y)
    return model