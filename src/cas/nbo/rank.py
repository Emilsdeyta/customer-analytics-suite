"""Combine propensity scores and association-rule lift into a top-N ranking."""
from __future__ import annotations

import pandas as pd


def rule_lifts_for_customer(purchased_products: set[str], candidate_products: list[str], rules_df: pd.DataFrame) -> dict[str, float]:
    """For each candidate product, find the best (max-lift) rule whose antecedents are
    a subset of what the customer already bought and whose consequents include the candidate.
    Candidates with no matching rule default to a neutral lift of 1.0.
    """
    lifts = {p: 1.0 for p in candidate_products}
    if rules_df is None or rules_df.empty:
        return lifts

    for _, rule in rules_df.iterrows():
        antecedents = set(rule["antecedents"])
        if not antecedents.issubset(purchased_products):
            continue
        for consequent in rule["consequents"]:
            if consequent in lifts:
                lifts[consequent] = max(lifts[consequent], float(rule["lift"]))
    return lifts


def hybrid_rank(propensity_scores: list[dict], rule_lifts: dict[str, float], top_n: int = 3) -> list[dict]:
    """rule_lifts: {product_name: lift_value}. Missing products default to lift=1.0 (neutral)."""
    ranked = []
    for item in propensity_scores:
        lift = rule_lifts.get(item["product"], 1.0)
        combined = item["propensity"] * lift
        ranked.append({**item, "lift": lift, "combined_score": combined})
    return sorted(ranked, key=lambda d: d["combined_score"], reverse=True)[:top_n]