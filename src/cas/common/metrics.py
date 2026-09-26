"""Shared evaluation metrics used across churn / nbo / clv modules."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def classification_report_dict(y_true, y_prob, k: float = 0.1) -> dict:
    """Return ROC-AUC, PR-AUC and top-decile lift for a binary classifier."""
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    order = np.argsort(y_prob)[::-1]
    top_n = max(1, int(len(y_true) * k))
    top_idx = order[:top_n]
    base_rate = np.mean(y_true)
    top_rate = np.mean(np.asarray(y_true)[top_idx])
    lift = top_rate / base_rate if base_rate > 0 else float("nan")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        f"top_{int(k * 100)}pct_lift": lift,
    }
