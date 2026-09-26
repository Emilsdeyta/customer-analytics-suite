import pandas as pd

from cas.clv.rfm import compute_rfm, segment_customers


def test_compute_rfm_basic():
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-15"]),
        "amount": [100, 50, 200],
    })
    rfm = compute_rfm(df, "customer_id", "date", "amount")
    assert set(rfm.columns) >= {"customer_id", "recency", "frequency", "monetary"}
    assert rfm.loc[rfm["customer_id"] == 1, "frequency"].iloc[0] == 2


def test_segment_customers_assigns_labels():
    rfm = pd.DataFrame({
        "customer_id": range(20),
        "recency": range(20),
        "frequency": range(1, 21),
        "monetary": range(100, 2100, 100),
    })
    segmented = segment_customers(rfm)
    assert "segment" in segmented.columns
