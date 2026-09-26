"""Inference for NBO: hybrid propensity x association-rule ranking for a known customer.

Unlike churn (where the caller supplies a feature dict), NBO recommendations are based on
purchase history the system already has — so the API only needs a customer_id, and looks up
that customer's precomputed features / purchase matrix from the training artifact.
"""
from __future__ import annotations

from cas.common.io import load_artifact
from cas.nbo.rank import hybrid_rank, rule_lifts_for_customer


def recommend_for_customer(customer_id: str, artifact_path: str = "models/nbo_model.joblib", top_n: int = 3) -> list[dict]:
    artifact = load_artifact(artifact_path)
    models = artifact["models"]
    rules_df = artifact["rules_df"]
    product_codes = artifact["product_codes"]
    descriptions = artifact["product_descriptions"]
    X = artifact["customer_features"]
    product_matrix = artifact["customer_product_matrix"]

    if customer_id not in X.index:
        raise KeyError(f"Unknown customer_id: {customer_id}")

    already = set(product_matrix.columns[product_matrix.loc[customer_id] == 1]) if customer_id in product_matrix.index else set()
    candidates = [p for p in product_codes if p not in already and p in models]
    if not candidates:
        return []

    x_row = X.loc[[customer_id]]
    scores = [{"product": p, "propensity": float(models[p].predict_proba(x_row)[:, 1][0])} for p in candidates]
    lifts = rule_lifts_for_customer(already, [s["product"] for s in scores], rules_df)
    ranked = hybrid_rank(scores, lifts, top_n=top_n)

    for r in ranked:
        r["description"] = descriptions.get(r["product"], "")
        r["reason"] = (
            "frequently bought with items you already own" if r["lift"] > 1.01 else "high individual propensity"
        )
    return ranked