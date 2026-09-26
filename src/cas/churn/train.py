"""CLI entrypoint: python -m cas.churn.train --config configs/churn.yaml"""
from __future__ import annotations

import argparse

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from cas.churn.data import clean, load_raw
from cas.churn.features import build_features, get_feature_target
from cas.common.config import load_config
from cas.common.io import save_artifact
from cas.common.metrics import classification_report_dict


def run(config_path: str) -> dict:
    cfg = load_config(config_path)

    df = load_raw(cfg.data.raw_path)
    df = clean(df)
    df = build_features(df)

    X, y = get_feature_target(df, cfg.data.target_column, cfg.data.id_column)
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_seed, stratify=y
    )

    model = LGBMClassifier(**cfg.model.params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = classification_report_dict(y_test, y_prob)

    save_artifact({"model": model, "columns": list(X.columns)}, f"{cfg.artifacts_dir}/churn_model.joblib")
    print("Churn model metrics:", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)
