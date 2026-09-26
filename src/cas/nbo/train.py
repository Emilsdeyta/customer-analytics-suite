"""CLI entrypoint: python -m cas.nbo.train --config configs/nbo.yaml

Hybrid Next-Best-Offer model:
  A) Propensity: one LightGBM classifier per top-N product, predicting P(customer buys product X)
     from RFM-style customer features.
  B) Association rules: FP-Growth mined over a bounded "candidate universe" of products
     (rules_universe_size) — mining over the full multi-thousand-SKU catalog is not memory-safe,
     so we bound it to the most frequent products, which is where cross-sell signal is strongest anyway.
  C) Collaborative filtering: SVD-based matrix factorization over the customer x product
     interaction matrix, which learns product-specific latent structure that (A) cannot.

Evaluation uses a TEMPORAL split (train on data up to a cutoff date, evaluate on the holdout period
that follows) rather than a random split, since recommending a product a customer already bought
before the cutoff would leak information.
"""
from __future__ import annotations

import argparse

import pandas as pd

from cas.common.config import load_config
from cas.common.io import save_artifact
from cas.nbo.cf import build_interaction_matrix, fit_svd, score_customer
from cas.nbo.data import clean, load_raw
from cas.nbo.features import (
    build_customer_features,
    build_customer_product_matrix,
    build_transactions,
    top_products,
)
from cas.nbo.propensity import train_propensity_models
from cas.nbo.rank import hybrid_rank, rule_lifts_for_customer
from cas.nbo.rules import mine_rules

FEATURE_COLUMNS = [
    "recency_days", "tenure_days", "n_invoices", "n_items",
    "n_distinct_products", "total_spent", "avg_basket_value", "avg_days_between_orders",
]


def run(config_path: str) -> dict:
    cfg = load_config(config_path)
    params = cfg.model.params

    print("Loading raw data (this can take a minute for ~1M rows)...")
    df = load_raw(cfg.data.raw_path)
    df = clean(df)

    holdout_days = params.get("holdout_days", 60)
    cutoff = df["InvoiceDate"].max() - pd.Timedelta(days=holdout_days)
    train_df = df[df["InvoiceDate"] <= cutoff]
    test_df = df[df["InvoiceDate"] > cutoff]
    print(f"Train rows: {len(train_df):,} | Holdout rows: {len(test_df):,} (cutoff={cutoff.date()})")

    top_n = params.get("top_n_products", 30)
    products = top_products(train_df, n=top_n)
    product_codes = products["StockCode"].tolist()
    product_descriptions = dict(zip(products["StockCode"], products["description"], strict=True))

    print("Building customer features...")
    customer_feats = build_customer_features(train_df)
    X = customer_feats.set_index("Customer ID")[FEATURE_COLUMNS]

    product_matrix = build_customer_product_matrix(train_df, product_codes).reindex(X.index, fill_value=0)
    y_dict = {code: product_matrix[code] for code in product_codes}

    print(f"Training propensity models for up to {top_n} products...")
    models = train_propensity_models(
        X, y_dict, min_positives=params.get("min_positives", 15), random_seed=cfg.data.random_seed
    )
    print(f"  -> {len(models)} products had enough positives to model")

    print("Mining association rules...")
    universe_size = params.get("rules_universe_size", 200)
    universe = top_products(train_df, n=universe_size)["StockCode"].tolist()
    bounded_df = train_df[train_df["StockCode"].isin(universe)]
    transactions = build_transactions(bounded_df)
    rules_df = mine_rules(
        transactions,
        min_support=params.get("min_support", 0.02),
        min_confidence=params.get("min_confidence", 0.3),
    )
    print(f"  -> {len(rules_df)} rules found (universe={universe_size} products)")

    print("Fitting collaborative filtering (SVD) model...")
    cf_matrix = build_interaction_matrix(train_df, product_codes).reindex(X.index, fill_value=0)
    cf_user_factors, cf_item_factors, _ = fit_svd(
        cf_matrix, n_components=params.get("cf_components", 15), random_seed=cfg.data.random_seed
    )
    cf_row_of = {cust: i for i, cust in enumerate(cf_matrix.index)}

    # --- Offline evaluation: hit rate@3 on the temporal holdout ---
    print("Evaluating hit rate@3 on holdout period...")
    test_purchases = test_df.groupby("Customer ID")["StockCode"].apply(set)
    eval_customers = [c for c in test_purchases.index if c in X.index]

    hits_hybrid, hits_propensity_only, hits_rules_only, hits_cf, total = 0, 0, 0, 0, 0
    for cust in eval_customers:
        already = set(product_matrix.columns[product_matrix.loc[cust] == 1]) if cust in product_matrix.index else set()
        candidates = [p for p in product_codes if p not in already and p in models]
        if not candidates:
            continue
        actual = test_purchases[cust]
        x_row = X.loc[[cust]]
        scores = [{"product": p, "propensity": float(models[p].predict_proba(x_row)[:, 1][0])} for p in candidates]
        lifts = rule_lifts_for_customer(already, [s["product"] for s in scores], rules_df)

        # Hybrid: propensity x lift
        hybrid_top3 = {r["product"] for r in hybrid_rank(scores, lifts, top_n=3)}
        # Propensity-only: ignore rules entirely (lift forced to neutral 1.0)
        propensity_top3 = {r["product"] for r in hybrid_rank(scores, {p: 1.0 for p in lifts}, top_n=3)}
        # Rules-only: ignore propensity entirely (score = lift alone)
        rules_scores = [{"product": s["product"], "propensity": 1.0} for s in scores]
        rules_top3 = {r["product"] for r in hybrid_rank(rules_scores, lifts, top_n=3)}

        hits_hybrid += bool(hybrid_top3 & actual)
        hits_propensity_only += bool(propensity_top3 & actual)
        hits_rules_only += bool(rules_top3 & actual)

        # Collaborative filtering (SVD): rank candidates by reconstructed latent-factor score
        if cust in cf_row_of:
            cf_scores = score_customer(cf_user_factors[cf_row_of[cust]], cf_item_factors, product_codes)
            cf_candidates = {p: s for p, s in cf_scores.items() if p not in already}
            cf_top3 = set(sorted(cf_candidates, key=cf_candidates.get, reverse=True)[:3])
            hits_cf += bool(cf_top3 & actual)

        total += 1

    hit_rate_at_3 = hits_hybrid / total if total else float("nan")
    hit_rate_propensity_only = hits_propensity_only / total if total else float("nan")
    hit_rate_rules_only = hits_rules_only / total if total else float("nan")
    hit_rate_cf = hits_cf / total if total else float("nan")

    # --- Baseline: recommend the 3 most popular products EXCLUDING items the customer
    # already owns (same rule the hybrid model follows) — otherwise the comparison is unfair,
    # since repeat-buyers (common among the wholesalers in this dataset) would trivially inflate it.
    print("Computing popularity baseline for comparison...")
    popularity_rank = products.sort_values("total_qty", ascending=False)["StockCode"].tolist()
    baseline_hits = 0
    for cust in eval_customers:
        already = set(product_matrix.columns[product_matrix.loc[cust] == 1]) if cust in product_matrix.index else set()
        baseline_rec = [p for p in popularity_rank if p not in already][:3]
        if set(baseline_rec) & test_purchases[cust]:
            baseline_hits += 1
    baseline_hit_rate_at_3 = baseline_hits / len(eval_customers) if eval_customers else float("nan")

    save_artifact(
        {
            "models": models,
            "rules_df": rules_df,
            "product_codes": product_codes,
            "product_descriptions": product_descriptions,
            "feature_columns": FEATURE_COLUMNS,
            "customer_features": X,
            "customer_product_matrix": product_matrix,
            "cf_user_factors": cf_user_factors,
            "cf_item_factors": cf_item_factors,
            "cf_customer_index": list(cf_matrix.index),
        },
        f"{cfg.artifacts_dir}/nbo_model.joblib",
    )

    metrics = {
        "hit_rate_at_3_hybrid": round(hit_rate_at_3, 4),
        "hit_rate_at_3_propensity_only": round(hit_rate_propensity_only, 4),
        "hit_rate_at_3_rules_only": round(hit_rate_rules_only, 4),
        "hit_rate_at_3_collaborative_filtering": round(hit_rate_cf, 4),
        "baseline_hit_rate_at_3": round(baseline_hit_rate_at_3, 4),
        "cf_lift_over_baseline": round(hit_rate_cf / baseline_hit_rate_at_3, 2) if baseline_hit_rate_at_3 else float("nan"),
        "hybrid_lift_over_baseline": round(hit_rate_at_3 / baseline_hit_rate_at_3, 2) if baseline_hit_rate_at_3 else float("nan"),
        "n_products_modeled": len(models),
        "n_rules": len(rules_df),
        "eval_customers": total,
    }
    print("NBO model metrics:", metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)