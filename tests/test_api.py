from fastapi.testclient import TestClient

import cas.api.routers.churn as churn_router
import cas.api.routers.clv as clv_router
import cas.api.routers.nbo as nbo_router
import cas.api.routers.priority as priority_router
from cas.api.main import app
from cas.scoring.priority import PriorityScoreResult

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_clv_endpoint_no_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(clv_router, "MODEL_PATH", str(tmp_path / "does_not_exist.joblib"))
    resp = client.post("/predict/clv", json={"customer_id": "C1", "recency": 10, "frequency": 5, "monetary": 200})
    assert resp.status_code == 503


def test_clv_endpoint_with_real_model(monkeypatch, clv_artifact_path):
    monkeypatch.setattr(clv_router, "MODEL_PATH", clv_artifact_path)
    resp = client.post("/predict/clv", json={"customer_id": "C1", "recency": 10, "frequency": 5, "monetary": 200})
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == "C1"
    assert body["predicted_clv"] >= 0.0


def test_churn_endpoint_with_real_model(monkeypatch, churn_artifact_path):
    path, feature_dict = churn_artifact_path
    monkeypatch.setattr(churn_router, "MODEL_PATH", path)
    resp = client.post("/predict/churn", json={"customer_id": "C1", "features": feature_dict})
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_nbo_endpoint_with_real_model(monkeypatch, nbo_artifact_path):
    monkeypatch.setattr(nbo_router, "MODEL_PATH", nbo_artifact_path)
    resp = client.post("/predict/nbo", json={"customer_id": "C200"})
    assert resp.status_code == 200
    assert isinstance(resp.json()["recommended_offers"], list)


def test_nbo_endpoint_unknown_customer(monkeypatch, nbo_artifact_path):
    monkeypatch.setattr(nbo_router, "MODEL_PATH", nbo_artifact_path)
    resp = client.post("/predict/nbo", json={"customer_id": "does-not-exist"})
    assert resp.status_code == 404


def test_priority_endpoint_success(monkeypatch):
    def fake_score(**kwargs):
        return PriorityScoreResult(
            priority_score=42.0, churn_probability=0.5, clv=100.0,
            best_offer_propensity=0.3, best_offer_product="P1",
        )

    monkeypatch.setattr(priority_router, "score_customer_pair", fake_score)
    resp = client.post(
        "/score/priority",
        json={
            "churn_customer_id": "c1", "churn_features": {},
            "retail_customer_id": "r1", "retail_recency": 1,
            "retail_frequency": 1, "retail_monetary": 1,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["priority_score"] == 42.0


def test_priority_endpoint_unknown_customer(monkeypatch):
    def fake_score(**kwargs):
        raise KeyError("unknown")

    monkeypatch.setattr(priority_router, "score_customer_pair", fake_score)
    resp = client.post(
        "/score/priority",
        json={
            "churn_customer_id": "c1", "churn_features": {},
            "retail_customer_id": "does-not-exist", "retail_recency": 1,
            "retail_frequency": 1, "retail_monetary": 1,
        },
    )
    assert resp.status_code == 404