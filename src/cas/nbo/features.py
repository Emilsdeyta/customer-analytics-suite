"""Feature engineering for NBO: customer-level features, top-N products, basket matrix."""
from __future__ import annotations

import pandas as pd


def top_products(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """Return the top-n products by total quantity sold, with a representative description."""
    agg = (
        df.groupby("StockCode")
        .agg(total_qty=("Quantity", "sum"), description=("Description", "first"))
        .sort_values("total_qty", ascending=False)
        .head(n)
        .reset_index()
    )
    return agg


def build_customer_features(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    """RFM-style customer features used as X for propensity models."""
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    grouped = df.groupby("Customer ID")
    feats = grouped.agg(
        recency_days=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        tenure_days=("InvoiceDate", lambda x: (x.max() - x.min()).days),
        n_invoices=("Invoice", "nunique"),
        n_items=("Quantity", "sum"),
        n_distinct_products=("StockCode", "nunique"),
        total_spent=("TotalPrice", "sum"),
    ).reset_index()
    feats["avg_basket_value"] = feats["total_spent"] / feats["n_invoices"].replace(0, 1)
    feats["avg_days_between_orders"] = feats["tenure_days"] / feats["n_invoices"].replace(0, 1)
    return feats


def build_customer_product_matrix(df: pd.DataFrame, product_codes: list[str]) -> pd.DataFrame:
    """Binary matrix: rows = Customer ID, columns = product_codes, 1 if ever purchased."""
    subset = df[df["StockCode"].isin(product_codes)]
    matrix = (
        subset.groupby(["Customer ID", "StockCode"]).size().unstack(fill_value=0)
        .clip(upper=1)
        .reindex(columns=product_codes, fill_value=0)
    )
    return matrix


def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    """One basket (list of StockCodes) per Invoice — input for association-rule mining."""
    return df.groupby("Invoice")["StockCode"].apply(list).tolist()