"""Cleaning, type normalization and categorical encoding for healthcare tables."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns with > this percentage of nulls are dropped.
HIGH_NULL_THRESHOLD = 0.80

# Columns whose names end with these suffixes are parsed as dates.
_DATE_SUFFIXES = ("Dt", "Date", "DATE")

# Binary categorical mappings applied when the column exists.
_BINARY_MAPS: dict[str, dict] = {
    # CMS Medicare encodes gender as 1=Male, 2=Female.
    "Gender": {1: 0, 2: 1},
    # Chronic condition flags: 1=Yes, 2=No in CMS data.
    "ChronicCond_Alzheimer": {1: 1, 2: 0},
    "ChronicCond_Heartfailure": {1: 1, 2: 0},
    "ChronicCond_KidneyDisease": {1: 1, 2: 0},
    "ChronicCond_Cancer": {1: 1, 2: 0},
    "ChronicCond_ObstrPulmonary": {1: 1, 2: 0},
    "ChronicCond_Depression": {1: 1, 2: 0},
    "ChronicCond_Diabetes": {1: 1, 2: 0},
    "ChronicCond_IschemicHeart": {1: 1, 2: 0},
    "ChronicCond_Osteoporasis": {1: 1, 2: 0},
    "ChronicCond_rheumatoidarthritis": {1: 1, 2: 0},
    "ChronicCond_stroke": {1: 1, 2: 0},
    "RenalDiseaseIndicator": {"Y": 1, "0": 0},
    "PotentialFraud": {"Yes": 1, "No": 0},
}

_MONEY_COLUMNS = [
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt",
]


def _drop_high_null_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    null_ratios = df.isnull().mean()
    cols_to_drop = null_ratios[null_ratios > HIGH_NULL_THRESHOLD].index.tolist()
    if cols_to_drop:
        logger.info("[%s] Dropping high-null columns: %s", name, cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [col for col in df.columns if col.endswith(_DATE_SUFFIXES)]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    if date_cols:
        logger.debug("Parsed date columns: %s", date_cols)
    return df


def _apply_binary_maps(df: pd.DataFrame) -> pd.DataFrame:
    for col, mapping in _BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df


def _cast_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = [col for col in _MONEY_COLUMNS if col in df.columns]
    if present:
        df[present] = df[present].astype("float32")
    return df


def clean_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a cleaned copy of df with dates parsed, categoricals encoded and types cast."""
    logger.info("[%s] Cleaning %s rows × %s columns", name, *df.shape)
    result = df.copy()

    result = _drop_high_null_columns(result, name)
    result = _parse_date_columns(result)
    result = _apply_binary_maps(result)
    result = _cast_money_columns(result)

    logger.info("[%s] Clean complete → %s rows × %s columns", name, *result.shape)
    return result
