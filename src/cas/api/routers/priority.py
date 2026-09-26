from fastapi import APIRouter, HTTPException

from cas.api.schemas import PriorityScoreRequest, PriorityScoreResponse
from cas.scoring.priority import score_customer_pair

router = APIRouter(prefix="/score", tags=["priority"])


@router.post("/priority", response_model=PriorityScoreResponse)
def get_priority_score(request: PriorityScoreRequest) -> PriorityScoreResponse:
    try:
        result = score_customer_pair(
            churn_customer_id=request.churn_customer_id,
            churn_features=request.churn_features,
            retail_customer_id=request.retail_customer_id,
            retail_recency=request.retail_recency,
            retail_frequency=request.retail_frequency,
            retail_monetary=request.retail_monetary,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Müştəri tapılmadı: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PriorityScoreResponse(
        priority_score=result.priority_score,
        churn_probability=result.churn_probability,
        clv=result.clv,
        best_offer_propensity=result.best_offer_propensity,
        best_offer_product=result.best_offer_product,
    )