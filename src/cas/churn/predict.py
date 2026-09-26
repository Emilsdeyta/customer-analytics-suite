"""Inference helpers for the churn model."""
from __future__ import annotations

import pandas as pd

from cas.churn.features import build_features
from cas.common.io import load_artifact


def predict(df: pd.DataFrame, artifact_path: str = "models/churn_model.joblib") -> pd.Series:
    """df: raw customer rows (same shape as training data, before encoding)."""
    artifact = load_artifact(artifact_path)
    model, columns = artifact["model"], artifact["columns"]
    df = build_features(df)
    X = pd.get_dummies(df, drop_first=True).reindex(columns=columns, fill_value=0)
    return pd.Series(model.predict_proba(X)[:, 1], index=df.index, name="churn_probability")


def predict_one(features: dict, artifact_path: str = "models/churn_model.joblib") -> tuple[float, pd.DataFrame]:
    """Convenience wrapper for a single customer dict (used by the API).

    Returns (probability, encoded_row) — the encoded_row is needed by explain.py for SHAP.
    """
    artifact = load_artifact(artifact_path)
    model, columns = artifact["model"], artifact["columns"]
    df = pd.DataFrame([features])
    df = build_features(df)
    X = pd.get_dummies(df, drop_first=True).reindex(columns=columns, fill_value=0)
    prob = float(model.predict_proba(X)[:, 1][0])
    return prob, X