"""Inferencia batch sobre DataFrame de proveedores."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from healthcare_fraud.features.preprocess import FEATURE_COLS


def predict_batch(pipeline: Pipeline, providers_df: pd.DataFrame) -> pd.DataFrame:
    """Aplica el pipeline completo y devuelve predicción y probabilidad por proveedor."""
    X = providers_df[FEATURE_COLS].to_numpy()
    probabilities: np.ndarray = pipeline.predict_proba(X)[:, 1]
    predictions: np.ndarray = (probabilities >= 0.5).astype(int)
    return pd.DataFrame(
        {"prediction": predictions, "probability_fraud": probabilities},
        index=providers_df.index,
    )
