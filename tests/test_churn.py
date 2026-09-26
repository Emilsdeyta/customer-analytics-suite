import pandas as pd

from cas.churn.data import clean
from cas.churn.features import build_features, get_feature_target
from cas.common.metrics import classification_report_dict


def test_clean_handles_bad_total_charges():
    df = pd.DataFrame({"customerID": ["1", "2"], "TotalCharges": [" ", "100.5"]})
    out = clean(df)
    assert out["TotalCharges"].isna().sum() == 0


def test_build_features_adds_columns():
    df = pd.DataFrame({"MonthlyCharges": [50], "tenure": [10], "TotalCharges": [500]})
    out = build_features(df)
    assert "avg_monthly_spend" in out.columns


def test_get_feature_target_splits_correctly():
    df = pd.DataFrame({"a": [1, 2], "Churn": ["Yes", "No"], "customerID": ["x", "y"]})
    X, y = get_feature_target(df, "Churn", "customerID")
    assert list(y) == [1, 0]
    assert "Churn" not in X.columns


def test_classification_report_dict_keys():
    y_true = [0, 1, 0, 1, 1]
    y_prob = [0.1, 0.8, 0.2, 0.6, 0.9]
    report = classification_report_dict(y_true, y_prob)
    assert "roc_auc" in report and "pr_auc" in report
