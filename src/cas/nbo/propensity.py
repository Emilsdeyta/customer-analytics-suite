"""Per-product propensity models (one-vs-rest LightGBM classifiers)."""
from __future__ import annotations

from lightgbm import LGBMClassifier


def train_propensity_models(X, y_dict: dict, min_positives: int = 15, random_seed: int = 42) -> dict:
    """y_dict: {product_code: binary_target_series} aligned to X's index.

    Products with fewer than `min_positives` positive examples are skipped —
    not enough signal to fit a meaningful classifier.
    """
    models = {}
    for product, y in y_dict.items():
        if y.sum() < min_positives:
            continue
        model = LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=random_seed, verbosity=-1)
        model.fit(X, y)
        models[product] = model
    return models