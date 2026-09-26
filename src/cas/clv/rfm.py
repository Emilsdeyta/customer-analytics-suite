"""RFM (Recency, Frequency, Monetary) feature construction and segmentation."""
from __future__ import annotations

import pandas as pd


def compute_rfm(df: pd.DataFrame, customer_col: str, date_col: str, amount_col: str, snapshot_date=None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg(
        recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(amount_col, "sum"),
    ).reset_index()
    return rfm


def segment_customers(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def label(score):
        if score >= 10:
            return "Champions"
        if score >= 8:
            return "Loyal"
        if score >= 6:
            return "At Risk"
        return "Hibernating"

    rfm["segment"] = rfm["rfm_score"].apply(label)
    return rfm
