"""Schema, type and business-rule validation for healthcare fraud tables."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Contrato mínimo por tabla: falla temprano si el dataset Kaggle cambia de esquema.
_REQUIRED_COLUMNS: dict[str, list[str]] = {
    "beneficiary": ["BeneID", "DOB", "Gender", "Race", "State", "County"],
    "inpatient": ["BeneID", "ClaimID", "Provider", "InscClaimAmtReimbursed"],
    "outpatient": ["BeneID", "ClaimID", "Provider", "InscClaimAmtReimbursed"],
    "labels_train": ["Provider", "PotentialFraud"],
    "labels_test": ["Provider"],
    # Tabla única tipo Kaggle consolidado (columnas snake_case del CSV publicado)
    "claims_flat": ["Provider_ID", "Claim_ID", "Is_Fraud"],
}

# Tables that contain reimbursement amount columns to check for negatives.
_AMOUNT_COLUMN = "InscClaimAmtReimbursed"

NULL_PCT_WARN_THRESHOLD = 30.0


def _check_required_columns(df: pd.DataFrame, name: str) -> None:
    required = _REQUIRED_COLUMNS.get(name, [])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing required columns: {missing}")


def _warn_high_nulls(df: pd.DataFrame, name: str) -> None:
    null_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    logger.info("[%s] Null percentage: %.2f%%", name, null_pct)
    if null_pct > NULL_PCT_WARN_THRESHOLD:
        logger.warning("[%s] High null percentage: %.2f%%", name, null_pct)


def _check_business_rules(df: pd.DataFrame, name: str) -> None:
    if _AMOUNT_COLUMN in df.columns:
        negative_count = (df[_AMOUNT_COLUMN] < 0).sum()
        if negative_count > 0:
            raise ValueError(f"[{name}] Found {negative_count} negative values in {_AMOUNT_COLUMN}")

    if "Claim_Amount" in df.columns:
        negative_count = (df["Claim_Amount"] < 0).sum()
        if negative_count > 0:
            raise ValueError(f"[{name}] Found {negative_count} negative values in Claim_Amount")

    if "PotentialFraud" in df.columns:
        valid_values = {"Yes", "No"}
        invalid = set(df["PotentialFraud"].dropna().unique()) - valid_values
        if invalid:
            raise ValueError(
                f"[{name}] Invalid PotentialFraud values: {invalid}. Expected {valid_values}"
            )

    if "Is_Fraud" in df.columns:
        invalid = set(df["Is_Fraud"].dropna().unique()) - {0, 1}
        if invalid:
            raise ValueError(f"[{name}] Invalid Is_Fraud values: {invalid}. Expected {{0, 1}}")


def validate_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate schema, null rates and business rules for a named table."""
    logger.info("[%s] Validating %s rows × %s columns", name, *df.shape)

    _check_required_columns(df, name)
    _warn_high_nulls(df, name)
    _check_business_rules(df, name)

    logger.info("[%s] Validation passed", name)
    return df
