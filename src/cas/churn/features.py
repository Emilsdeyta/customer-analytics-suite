"""Feature engineering for churn model.

IMPORTANT: split train/test BEFORE fitting any encoder to avoid leakage.
"""
from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"MonthlyCharges", "tenure"}.issubset(df.columns):
        df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
        df["spend_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_spend"] + 1e-6)
    return df


def get_feature_target(df: pd.DataFrame, target_col: str, id_col: str):
    col = df[target_col]
    if pd.api.types.is_numeric_dtype(col):
        y = col
    else:
        y = col.map({"Yes": 1, "No": 0})
    X = df.drop(columns=[target_col, id_col], errors="ignore")
    return X, y
