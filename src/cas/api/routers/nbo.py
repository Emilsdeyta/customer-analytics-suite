from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from cas.api.schemas import NBORequest, NBOResponse
from cas.nbo.predict import recommend_for_customer

router = APIRouter(prefix="/predict", tags=["nbo"])

MODEL_PATH = "models/nbo_model.joblib"


@router.post("/nbo", response_model=NBOResponse)
def predict_nbo(req: NBORequest) -> NBOResponse:
    if not Path(MODEL_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail="NBO model not trained yet. Run: python -m cas.nbo.train --config configs/nbo.yaml",
        )
    try:
        offers = recommend_for_customer(req.customer_id, artifact_path=MODEL_PATH)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown customer_id: {req.customer_id}")

    return NBOResponse(customer_id=req.customer_id, recommended_offers=offers)