"""Pruebas unitarias para entrenamiento tipo baseline (sin MLflow)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from healthcare_fraud.features.preprocess import FEATURE_COLS, build_preprocessing_pipeline
from healthcare_fraud.models.evaluate import evaluate_model

EXPECTED_METRIC_KEYS = frozenset({"recall", "precision", "f1", "roc_auc", "avg_precision"})


def train_baseline(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_val_raw: np.ndarray,
    y_val: np.ndarray,
    *,
    random_state: int = 42,
) -> dict[str, float]:
    """Pipeline imputación + escalado + regresión logística balanceada (baseline reproducible)."""
    preprocessor = build_preprocessing_pipeline()
    clf = LogisticRegression(
        max_iter=2000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    pipe = Pipeline([*preprocessor.steps, ("classifier", clf)])
    pipe.fit(X_train_raw, y_train)
    return evaluate_model(pipe, X_val_raw, y_val)


def test_train_baseline_returns_expected_metrics_shape_and_ranges() -> None:
    rng = np.random.default_rng(42)
    n_train = 160
    n_val = 40
    X_train_raw = rng.standard_normal((n_train, len(FEATURE_COLS)))
    X_val_raw = rng.standard_normal((n_val, len(FEATURE_COLS)))
    y_train = (rng.random(n_train) > 0.88).astype(np.int64)
    y_val = (rng.random(n_val) > 0.88).astype(np.int64)
    # Garantizar ambas clases para ROC-AUC estable
    y_train[0] = 0
    y_train[1] = 1
    y_val[0] = 0
    y_val[1] = 1

    metrics = train_baseline(X_train_raw, y_train, X_val_raw, y_val)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS
    for name, value in metrics.items():
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0, f"{name}={value} fuera de [0, 1]"


def test_train_baseline_metrics_deterministic_with_fixed_seed() -> None:
    """Con datos y semilla fijas las métricas son reproducibles."""
    rng = np.random.default_rng(12345)
    X_train_raw = rng.standard_normal((120, len(FEATURE_COLS)))
    X_val_raw = rng.standard_normal((30, len(FEATURE_COLS)))
    y_train = (rng.random(120) > 0.6).astype(np.int64)
    y_val = (rng.random(30) > 0.6).astype(np.int64)
    y_train[0], y_train[1] = 0, 1
    y_val[0], y_val[1] = 0, 1

    m1 = train_baseline(X_train_raw, y_train, X_val_raw, y_val, random_state=0)
    m2 = train_baseline(X_train_raw, y_train, X_val_raw, y_val, random_state=0)

    assert m1 == m2
