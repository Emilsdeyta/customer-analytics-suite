"""Probabilistic CLV: BG/NBD (purchase frequency) + Gamma-Gamma (monetary value)."""
from __future__ import annotations

import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter


def fit_btyd(rfm_lifetimes_df: pd.DataFrame):
    """Expects columns: frequency, recency, T, monetary_value (lifetimes package convention)."""
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm_lifetimes_df["frequency"], rfm_lifetimes_df["recency"], rfm_lifetimes_df["T"])

    ggf = GammaGammaFitter(penalizer_coef=0.01)
    paying = rfm_lifetimes_df[rfm_lifetimes_df["frequency"] > 0]
    ggf.fit(paying["frequency"], paying["monetary_value"])

    return bgf, ggf


def predict_clv(bgf, ggf, rfm_lifetimes_df: pd.DataFrame, months: int = 12, discount_rate: float = 0.01) -> pd.Series:
    return ggf.customer_lifetime_value(
        bgf,
        rfm_lifetimes_df["frequency"],
        rfm_lifetimes_df["recency"],
        rfm_lifetimes_df["T"],
        rfm_lifetimes_df["monetary_value"],
        time=months,
        discount_rate=discount_rate,
    )
