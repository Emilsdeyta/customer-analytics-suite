"""CLI entrypoint: python -m cas.clv.train --config configs/clv.yaml

Two competing approaches, evaluated head-to-head on the SAME temporal holdout:
  A) BG/NBD + Gamma-Gamma (lifetimes) — the probabilistic BTYD standard. Separately models
     "how many future transactions" and "what's the average order value", which tends to be
     more stable for customers with short histories ("new customer" cold-start).
  B) RFM -> XGBoost regression baseline — simpler, needs a historical holdout period to
     generate a training label, and can incorporate more features later if needed.

Evaluation: fit on transactions up to a calibration cutoff, then compare predicted CLV
(over the holdout horizon) against ACTUAL money spent by each customer in that holdout
period. Spearman correlation matters more than absolute error here, because in practice
CLV is used to RANK/prioritize customers, not to forecast an exact dollar figure.
"""
from __future__ import annotations

import argparse

import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from cas.clv.btyd import fit_btyd, predict_clv
from cas.clv.data import clean, load_raw
from cas.clv.forecast import train_baseline
from cas.clv.rfm import compute_rfm, segment_customers
from cas.common.config import load_config
from cas.common.io import save_artifact


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    params = cfg.model.params

    print("Loading raw data...")
    df = load_raw(cfg.data.raw_path)
    df = clean(df)

    holdout_days = params.get("holdout_days", 90)
    cutoff = df["InvoiceDate"].max() - pd.Timedelta(days=holdout_days)
    calib_df = df[df["InvoiceDate"] <= cutoff]
    holdout_df = df[df["InvoiceDate"] > cutoff]
    print(f"Calibration rows: {len(calib_df):,} | Holdout rows: {len(holdout_df):,} (cutoff={cutoff.date()})")

    print("Building BTYD summary (frequency, recency, T, monetary_value)...")
    summary = summary_data_from_transaction_data(
        calib_df,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=cutoff,
        freq="D",
    )

    print("Fitting BG/NBD + Gamma-Gamma...")
    repeat_customers = summary[summary["frequency"] > 0]
    bgf, ggf = fit_btyd(summary)
    predicted_clv = predict_clv(
        bgf, ggf, repeat_customers, months=holdout_days / 30, discount_rate=params.get("discount_rate", 0.0)
    )

    print("Computing actual holdout spend per customer (ground truth)...")
    actual_spend = holdout_df.groupby("Customer ID")["TotalPrice"].sum()

    rfm = compute_rfm(calib_df, "Customer ID", "InvoiceDate", "TotalPrice").set_index("Customer ID")

    eval_df = (
        pd.DataFrame({"predicted_clv": predicted_clv})
        .join(rfm[["recency", "frequency", "monetary"]], how="inner")
        .join(actual_spend.rename("actual_spend"), how="left")
    )
    eval_df["actual_spend"] = eval_df["actual_spend"].fillna(0)

    print(f"Evaluating on {len(eval_df)} customers with sufficient calibration history...")
    train_idx, test_idx = train_test_split(
        eval_df.index.to_numpy(dtype=object), test_size=0.3, random_state=cfg.data.random_seed
    )
    btyd_spearman, _ = spearmanr(eval_df.loc[test_idx, "predicted_clv"], eval_df.loc[test_idx, "actual_spend"])
    btyd_mae = mean_absolute_error(eval_df.loc[test_idx, "actual_spend"], eval_df.loc[test_idx, "predicted_clv"])

    print("Training XGBoost RFM baseline for comparison (same train/test split)...")
    feature_cols = ["recency", "frequency", "monetary"]
    xgb_model = train_baseline(
        eval_df.loc[train_idx, feature_cols], eval_df.loc[train_idx, "actual_spend"], random_seed=cfg.data.random_seed
    )
    xgb_pred = xgb_model.predict(eval_df.loc[test_idx, feature_cols])
    xgb_spearman, _ = spearmanr(xgb_pred, eval_df.loc[test_idx, "actual_spend"])
    xgb_mae = mean_absolute_error(eval_df.loc[test_idx, "actual_spend"], xgb_pred)

    print("Segmenting the full customer base (RFM quartiles)...")
    full_rfm = compute_rfm(df, "Customer ID", "InvoiceDate", "TotalPrice")
    segmented = segment_customers(full_rfm)

    save_artifact(
        {
            "bgf": bgf,
            "ggf": ggf,
            "summary": summary,
            "segmented_customers": segmented,
            "xgb_baseline": xgb_model,
            "feature_cols": feature_cols,
        },
        f"{cfg.artifacts_dir}/clv_model.joblib",
    )

    metrics = {
        "btyd_spearman": round(float(btyd_spearman), 4),
        "btyd_mae": round(float(btyd_mae), 2),
        "xgb_baseline_spearman": round(float(xgb_spearman), 4),
        "xgb_baseline_mae": round(float(xgb_mae), 2),
        "n_customers_evaluated": len(test_idx),
    }
    print("CLV model metrics:", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)