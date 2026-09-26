"""Shared I/O helpers: reading raw data, saving/loading model artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    """Read csv/parquet based on file extension."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_artifact(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str | Path) -> Any:
    return joblib.load(path)
