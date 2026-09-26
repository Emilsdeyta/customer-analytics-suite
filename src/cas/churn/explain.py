"""SHAP-based explainability for the churn model (global + per-customer)."""
from __future__ import annotations

import shap

from cas.common.io import load_artifact


def top_drivers_for_customer(X_row, artifact_path: str = "models/churn_model.joblib", top_n: int = 3):
    """X_row: a single-row, already-encoded DataFrame (as returned by predict.predict_one)."""
    artifact = load_artifact(artifact_path)
    model = artifact["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_row)
    # Binary classifiers can return either a single array or a list [class0, class1]
    values = shap_values[1] if isinstance(shap_values, list) else shap_values
    contributions = sorted(
        zip(X_row.columns, values[0]), key=lambda t: abs(t[1]), reverse=True
    )[:top_n]
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in contributions]