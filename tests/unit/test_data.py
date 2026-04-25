"""Unit tests for data load, validate and clean modules."""

from __future__ import annotations

import pandas as pd
import pytest

from healthcare_fraud.data.clean import clean_dataframe
from healthcare_fraud.data.validate import validate_dataframe

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def beneficiary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BeneID": ["B001", "B002", "B003"],
            "DOB": ["1950-01-01", "1960-06-15", "1945-03-20"],
            "Gender": [1, 2, 1],
            "Race": [1, 2, 3],
            "State": [10, 20, 30],
            "County": [100, 200, 300],
        }
    )


@pytest.fixture
def inpatient_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BeneID": ["B001", "B002"],
            "ClaimID": ["C001", "C002"],
            "Provider": ["P001", "P002"],
            "InscClaimAmtReimbursed": [500.0, 1200.0],
        }
    )


@pytest.fixture
def labels_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Provider": ["P001", "P002", "P003"],
            "PotentialFraud": ["Yes", "No", "Yes"],
        }
    )


# ---------------------------------------------------------------------------
# validate_dataframe
# ---------------------------------------------------------------------------


def test_validate_beneficiary_passes(beneficiary_df: pd.DataFrame) -> None:
    result = validate_dataframe(beneficiary_df, "beneficiary")
    assert result is not None
    assert len(result) == len(beneficiary_df)


def test_validate_raises_on_missing_bene_column(beneficiary_df: pd.DataFrame) -> None:
    df_bad = beneficiary_df.drop(columns=["BeneID"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df_bad, "beneficiary")


def test_validate_inpatient_passes(inpatient_df: pd.DataFrame) -> None:
    result = validate_dataframe(inpatient_df, "inpatient")
    assert result is not None


def test_validate_raises_on_negative_amount(inpatient_df: pd.DataFrame) -> None:
    df_bad = inpatient_df.copy()
    df_bad.loc[0, "InscClaimAmtReimbursed"] = -100.0
    with pytest.raises(ValueError, match="negative values"):
        validate_dataframe(df_bad, "inpatient")


def test_validate_labels_raises_on_invalid_fraud_value(
    labels_train_df: pd.DataFrame,
) -> None:
    df_bad = labels_train_df.copy()
    df_bad.loc[0, "PotentialFraud"] = "Maybe"
    with pytest.raises(ValueError, match="Invalid PotentialFraud"):
        validate_dataframe(df_bad, "labels_train")


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------


def test_clean_encodes_gender(beneficiary_df: pd.DataFrame) -> None:
    cleaned = clean_dataframe(beneficiary_df, "beneficiary")
    assert set(cleaned["Gender"].dropna().unique()).issubset({0, 1})


def test_clean_encodes_fraud_labels(labels_train_df: pd.DataFrame) -> None:
    cleaned = clean_dataframe(labels_train_df, "labels_train")
    assert set(cleaned["PotentialFraud"].unique()) == {1, 0}


def test_clean_parses_date_columns(beneficiary_df: pd.DataFrame) -> None:
    cleaned = clean_dataframe(beneficiary_df, "beneficiary")
    assert pd.api.types.is_datetime64_any_dtype(cleaned["DOB"])


def test_clean_casts_money_to_float32(inpatient_df: pd.DataFrame) -> None:
    cleaned = clean_dataframe(inpatient_df, "inpatient")
    assert cleaned["InscClaimAmtReimbursed"].dtype == "float32"


def test_clean_drops_high_null_columns() -> None:
    df = pd.DataFrame(
        {
            "BeneID": ["B001", "B002", "B003"],
            "sparse_col": [None, None, None],
        }
    )
    cleaned = clean_dataframe(df, "beneficiary")
    assert "sparse_col" not in cleaned.columns


def test_clean_returns_copy(beneficiary_df: pd.DataFrame) -> None:
    cleaned = clean_dataframe(beneficiary_df, "beneficiary")
    assert cleaned is not beneficiary_df
