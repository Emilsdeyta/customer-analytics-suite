from fastapi.testclient import TestClient

from cas.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_clv_endpoint():
    resp = client.post("/predict/clv", json={
        "customer_id": "C1", "recency": 10, "frequency": 5, "monetary": 200
    })
    assert resp.status_code == 200
    assert "predicted_clv" in resp.json()
