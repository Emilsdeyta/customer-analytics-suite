"""Matrix-factorization based collaborative filtering for NBO (implicit feedback).

Unlike the propensity models (one classifier per product, fed only customer-level RFM
features), this approach learns latent factors for BOTH customers and products from the
customer x product interaction matrix directly — so it can capture "customers who buy
product A also tend to buy product C" patterns that customer-only features cannot see.

We use scikit-learn's TruncatedSVD (no extra native dependency, no Windows compile issues)
rather than a dedicated implicit-ALS library. It's a reasonable, well-understood baseline
for implicit-feedback matrix factorization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


def build_interaction_matrix(df: pd.DataFrame, product_codes: list[str]) -> pd.DataFrame:
    """Customer x product matrix of log1p(total quantity purchased) — implicit feedback strength."""
    subset = df[df["StockCode"].isin(product_codes)]
    qty = subset.groupby(["Customer ID", "StockCode"])["Quantity"].sum().unstack(fill_value=0)
    qty = qty.reindex(columns=product_codes, fill_value=0)
    return np.log1p(qty)


def fit_svd(matrix: pd.DataFrame, n_components: int = 15, random_seed: int = 42):
    """Fit truncated SVD on the interaction matrix. Returns (user_factors, item_factors, svd)."""
    n_components = max(2, min(n_components, min(matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_seed)
    user_factors = svd.fit_transform(matrix.values)  # (n_customers, k)
    item_factors = svd.components_  # (k, n_products)
    return user_factors, item_factors, svd


def score_customer(user_vector: np.ndarray, item_factors: np.ndarray, product_codes: list[str]) -> dict[str, float]:
    """Reconstructed affinity score for every product, for one customer's latent vector."""
    scores = user_vector @ item_factors
    return dict(zip(product_codes, scores))