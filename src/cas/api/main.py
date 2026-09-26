"""FastAPI application entrypoint.

Run locally:
    uvicorn cas.api.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI

from cas.api.routers import churn, clv, nbo, priority

app = FastAPI(
    title="Customer Analytics Suite API",
    description="Churn / Next-Best-Offer / CLV prediction endpoints",
    version="0.1.0",
)

app.include_router(churn.router)
app.include_router(nbo.router)
app.include_router(clv.router)
app.include_router(priority.router)


@app.get("/health")
def health():
    return {"status": "ok"}
