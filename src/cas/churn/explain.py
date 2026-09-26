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

    if isinstance(shap_values, list):
        # Older shap versions: list of per-class arrays, shape (n_samples, n_features) each.
        values = shap_values[1]
    elif shap_values.ndim == 3:
        # Newer shap versions: single array, shape (n_samples, n_features, n_classes).
        values = shap_values[:, :, 1]
    else:
        values = shap_values

    contributions = sorted(
        zip(X_row.columns, values[0], strict=True), key=lambda t: abs(t[1]), reverse=True
    )[:top_n]
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in contributions]