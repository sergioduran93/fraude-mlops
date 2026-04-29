"""Model evaluation metrics for binary fraud classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Return recall, precision, f1, roc_auc and avg_precision for binary classification."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "avg_precision": round(float(average_precision_score(y_true, y_prob)), 4),
    }
