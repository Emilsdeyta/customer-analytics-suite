"""Load and clean the Telco Customer Churn dataset (Kaggle/IBM sample)."""
from __future__ import annotations

import pandas as pd

from cas.common.io import read_table


def load_raw(path: str) -> pd.DataFrame:
    return read_table(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix TotalCharges dtype, drop empty ids, standardize Yes/No."""
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.dropna(subset=[c for c in ["customerID"] if c in df.columns])
    return df
