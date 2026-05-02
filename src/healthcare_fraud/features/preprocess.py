"""Split and preprocessing pipeline for provider-level feature matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from healthcare_fraud.config import SETTINGS

FEATURE_COLS: list[str] = [
    "total_claims",
    "ip_claims",
    "op_claims",
    "unique_benes",
    "unique_attending",
    "total_reimbursed",
    "mean_reimbursed",
    "max_reimbursed",
    "total_deductible",
    "mean_claim_duration",
    "mean_hosp_stay",
    "mean_bene_age",
    "pct_deceased",
    "mean_chronic_count",
    "claims_per_bene",
    "reimbursed_per_bene",
]


def split_providers(
    feature_df: pd.DataFrame,
    train_ratio: float = SETTINGS.train_ratio,
    random_state: int = SETTINGS.random_state,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partición train/val estratificada por proveedor (una fila = un proveedor).

    Estratificar por ``PotentialFraud`` mantiene la proporción de fraude en val;
    al agrupar por proveedor se reduce fuga de información entre conjuntos.
    """
    if "PotentialFraud" not in feature_df.columns:
        raise ValueError("feature_df must contain 'PotentialFraud' column")
    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing:
        raise ValueError(f"feature_df is missing feature columns: {missing}")
    train_df, val_df = train_test_split(
        feature_df,
        test_size=1 - train_ratio,
        random_state=random_state,
        stratify=feature_df["PotentialFraud"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_preprocessing_pipeline() -> Pipeline:
    """Imputation + scaling pipeline (no classifier — added in train.py)."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def prepare_train_val(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """Ajusta imputación y escalado solo con train; val solo se transforma.

    Los arrays escalados alimentan Optuna/XGBoost; las matrices *raw* se usan con el
    pipeline completo en el entrenamiento final (mismo orden que ``FEATURE_COLS``).
    """
    X_train = train_df[FEATURE_COLS].values
    X_val = val_df[FEATURE_COLS].values
    y_train = train_df["PotentialFraud"].values
    y_val = val_df["PotentialFraud"].values

    preprocessor = build_preprocessing_pipeline()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    return X_train_scaled, X_val_scaled, y_train, y_val, preprocessor
