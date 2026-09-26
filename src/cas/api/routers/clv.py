from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from cas.api.schemas import CLVRequest, CLVResponse
from cas.clv.predict import predict_clv as predict_clv_value

router = APIRouter(prefix="/predict", tags=["clv"])

MODEL_PATH = "models/clv_model.joblib"


@router.post("/clv", response_model=CLVResponse)
def predict_clv(req: CLVRequest) -> CLVResponse:
    if not Path(MODEL_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail="CLV model not trained yet. Run: python -m cas.clv.train --config configs/clv.yaml",
        )

    value = predict_clv_value(
        recency=req.recency,
        frequency=req.frequency,
        monetary=req.monetary,
        artifact_path=MODEL_PATH,
    )
    return CLVResponse(customer_id=req.customer_id, predicted_clv=round(value, 2))