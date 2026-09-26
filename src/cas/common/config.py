"""Central configuration loading via pydantic-settings + YAML.

Usage:
    from cas.common.config import load_config
    cfg = load_config("configs/churn.yaml")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    raw_path: str
    processed_path: str
    target_column: str
    id_column: str
    test_size: float = 0.2
    random_seed: int = 42


class ModelConfig(BaseModel):
    name: str
    params: dict[str, Any] = {}


class TrainConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    artifacts_dir: str = "models"


def load_config(path: str | Path) -> TrainConfig:
    """Load a YAML config file into a validated TrainConfig object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(**raw)
