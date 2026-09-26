"""Shared pytest fixtures: builds tiny, REAL (not mocked) trained artifacts so that
router/API tests exercise the actual inference code path (predict_one, explain,
recommend_for_customer, predict_clv) without needing the full multi-MB production
models or raw datasets, which are intentionally excluded from git (see .gitignore).
"""
from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from cas.churn.features import build_features, get_feature_target
from cas.common.io import save_artifact

CHURN_RAW_ROWS = [
    {
        "customerID": "1000-AAA", "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85, "TotalCharges": 29.85, "Churn": "No",
    },
    {
        "customerID": "1001-BBB", "gender": "Male", "SeniorCitizen": 1, "Partner": "No",
        "Dependents": "No", "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 105.5, "TotalCharges": 6330.0, "Churn": "Yes",
    },
    {
        "customerID": "1002-CCC", "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "Yes", "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "One year", "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.0, "TotalCharges": 660.0, "Churn": "No",
    },
    {
        "customerID": "1003-DDD", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "Yes", "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 90.0, "TotalCharges": 270.0, "Churn": "Yes",
    },
]


@pytest.fixture
def churn_artifact_path(tmp_path):
    """(path, feature_dict): a tiny RandomForest trained on 4 synthetic rows -- enough
    for shap.TreeExplainer + predict_proba to work end-to-end, no real Telco data needed."""
    raw = pd.DataFrame(CHURN_RAW_ROWS)
    built = build_features(raw)
    X, y = get_feature_target(built, target_col="Churn", id_col="customerID")
    X_encoded = pd.get_dummies(X, drop_first=True)

    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=0)
    model.fit(X_encoded, y)

    path = tmp_path / "churn_model.joblib"
    save_artifact({"model": model, "columns": list(X_encoded.columns)}, path)
    return str(path), X.iloc[0].to_dict()


@pytest.fixture
def clv_artifact_path(tmp_path):
    """Tiny XGBRegressor trained on synthetic RFM triples -- exercises the real
    cas.clv.predict.predict_clv() code path without the full Online Retail II dataset."""
    X = pd.DataFrame(
        {
            "recency": [5, 40, 100, 300, 10, 60],
            "frequency": [20, 8, 3, 1, 15, 5],
            "monetary": [2000, 500, 150, 20, 1500, 300],
        }
    )
    y = pd.Series([2200, 600, 100, 5, 1800, 250])

    model = XGBRegressor(n_estimators=10, max_depth=2, random_state=0)
    model.fit(X, y)

    path = tmp_path / "clv_model.joblib"
    save_artifact({"xgb_baseline": model, "feature_cols": ["recency", "frequency", "monetary"]}, path)
    return str(path)


@pytest.fixture
def nbo_artifact_path(tmp_path):
    """Tiny hybrid NBO artifact: 2 propensity models, empty rules table (neutral lift),
    2 customers, 2 candidate products -- exercises recommend_for_customer() end-to-end."""
    product_codes = ["P1", "P2"]
    customer_features = pd.DataFrame(
        {"recency": [5, 50], "frequency": [10, 2], "monetary": [500, 50]},
        index=["C100", "C200"],
    )
    customer_product_matrix = pd.DataFrame({"P1": [1, 0], "P2": [0, 0]}, index=["C100", "C200"])
    rules_df = pd.DataFrame(columns=["antecedents", "consequents", "lift"])

    models = {}
    for product in product_codes:
        y = pd.Series([1, 0], index=customer_features.index)
        m = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=0)
        m.fit(customer_features, y)
        models[product] = m

    path = tmp_path / "nbo_model.joblib"
    save_artifact(
        {
            "models": models,
            "rules_df": rules_df,
            "product_codes": product_codes,
            "product_descriptions": {"P1": "Widget", "P2": "Gadget"},
            "customer_features": customer_features,
            "customer_product_matrix": customer_product_matrix,
        },
        path,
    )
    return str(path)