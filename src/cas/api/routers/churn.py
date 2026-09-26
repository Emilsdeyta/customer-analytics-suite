from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from cas.api.schemas import ChurnRequest, ChurnResponse
from cas.churn.explain import top_drivers_for_customer
from cas.churn.predict import predict_one

router = APIRouter(prefix="/predict", tags=["churn"])

MODEL_PATH = "models/churn_model.joblib"


def _risk_tier(prob: float) -> str:
    if prob > 0.6:
        return "high"
    if prob > 0.3:
        return "medium"
    return "low"


@router.post("/churn", response_model=ChurnResponse)
def predict_churn(req: ChurnRequest) -> ChurnResponse:
    if not Path(MODEL_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail="Churn model not trained yet. Run: python -m cas.churn.train --config configs/churn.yaml",
        )

    prob, X_row = predict_one(req.features, artifact_path=MODEL_PATH)

    try:
        drivers = top_drivers_for_customer(X_row, artifact_path=MODEL_PATH)
    except Exception:
        drivers = []  # SHAP is best-effort; never fail the request because of it

    return ChurnResponse(
        customer_id=req.customer_id,
        churn_probability=round(prob, 4),
        risk_tier=_risk_tier(prob),
        top_drivers=drivers,
    )