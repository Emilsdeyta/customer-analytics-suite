"""Pydantic request/response schemas for the FastAPI layer."""
from __future__ import annotations

from pydantic import BaseModel


class ChurnRequest(BaseModel):
    customer_id: str
    features: dict


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_tier: str
    top_drivers: list[dict] = []


class NBORequest(BaseModel):
    customer_id: str
    # features: dict


class NBOResponse(BaseModel):
    customer_id: str
    recommended_offers: list[dict]


class CLVRequest(BaseModel):
    customer_id: str
    recency: float
    frequency: float
    monetary: float


class CLVResponse(BaseModel):
    customer_id: str
    predicted_clv: float


class PriorityScoreRequest(BaseModel):
    churn_customer_id: str
    churn_features: dict

    retail_customer_id: str
    retail_recency: float
    retail_frequency: float
    retail_monetary: float


class PriorityScoreResponse(BaseModel):
    priority_score: float
    churn_probability: float
    clv: float
    best_offer_propensity: float
    best_offer_product: str | None