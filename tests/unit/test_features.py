"""Unit tests for features/build and features/preprocess modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from healthcare_fraud.features.build import build_features
from healthcare_fraud.features.preprocess import (
    FEATURE_COLS,
    build_preprocessing_pipeline,
    prepare_train_val,
    split_providers,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def inpatient_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BeneID": ["B001", "B002", "B001", "B002"],
            "ClaimID": ["C001", "C002", "C003", "C004"],
            "Provider": ["P001", "P001", "P002", "P002"],
            "ClaimStartDt": pd.to_datetime(
                ["2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01"]
            ),
            "ClaimEndDt": pd.to_datetime(["2009-01-10", "2009-02-05", "2009-03-07", "2009-04-03"]),
            "InscClaimAmtReimbursed": np.array([500.0, 1200.0, 800.0, 600.0], dtype="float32"),
            "DeductibleAmtPaid": np.array([50.0, 100.0, 80.0, 60.0], dtype="float32"),
            "AttendingPhysician": ["DR001", "DR002", "DR003", "DR003"],
            "AdmissionDt": pd.to_datetime(["2009-01-01", "2009-02-01", "2009-03-01", "2009-04-01"]),
            "DischargeDt": pd.to_datetime(["2009-01-05", "2009-02-04", "2009-03-05", "2009-04-02"]),
        }
    )


@pytest.fixture
def outpatient_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BeneID": ["B001", "B002", "B001", "B002"],
            "ClaimID": ["O001", "O002", "O003", "O004"],
            "Provider": ["P001", "P001", "P002", "P002"],
            "ClaimStartDt": pd.to_datetime(
                ["2009-05-01", "2009-06-01", "2009-07-01", "2009-08-01"]
            ),
            "ClaimEndDt": pd.to_datetime(["2009-05-03", "2009-06-02", "2009-07-02", "2009-08-01"]),
            "InscClaimAmtReimbursed": np.array([200.0, 300.0, 150.0, 400.0], dtype="float32"),
            "DeductibleAmtPaid": np.array([20.0, 30.0, 15.0, 40.0], dtype="float32"),
            "AttendingPhysician": ["DR001", "DR002", "DR003", "DR004"],
        }
    )


@pytest.fixture
def beneficiary_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BeneID": ["B001", "B002"],
            "DOB": pd.to_datetime(["1950-01-01", "1960-06-15"]),
            "DOD": [pd.NaT, pd.NaT],
            "Gender": [0, 1],
            "Race": [1, 2],
            "State": [10, 20],
            "County": [100, 200],
            "NoOfMonths_PartACov": [12, 12],
            "NoOfMonths_PartBCov": [12, 10],
            "IPAnnualReimbursementAmt": np.array([2000.0, 1500.0], dtype="float32"),
            "IPAnnualDeductibleAmt": np.array([200.0, 150.0], dtype="float32"),
            "OPAnnualReimbursementAmt": np.array([500.0, 400.0], dtype="float32"),
            "OPAnnualDeductibleAmt": np.array([50.0, 40.0], dtype="float32"),
            "RenalDiseaseIndicator": [0, 1],
            "ChronicCond_Alzheimer": [0, 1],
            "ChronicCond_Heartfailure": [1, 0],
            "ChronicCond_KidneyDisease": [0, 0],
            "ChronicCond_Cancer": [0, 0],
            "ChronicCond_ObstrPulmonary": [0, 1],
            "ChronicCond_Depression": [1, 0],
            "ChronicCond_Diabetes": [0, 1],
            "ChronicCond_IschemicHeart": [0, 0],
            "ChronicCond_Osteoporasis": [0, 0],
            "ChronicCond_rheumatoidarthritis": [0, 0],
            "ChronicCond_stroke": [0, 0],
        }
    )


@pytest.fixture
def labels_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Provider": ["P001", "P002"],
            "PotentialFraud": [1, 0],
        }
    )


@pytest.fixture
def tables_dict(
    inpatient_df: pd.DataFrame,
    outpatient_df: pd.DataFrame,
    beneficiary_df: pd.DataFrame,
    labels_train_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "inpatient": inpatient_df,
        "outpatient": outpatient_df,
        "beneficiary": beneficiary_df,
        "labels_train": labels_train_df,
    }


@pytest.fixture
def feature_df(tables_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_features(tables_dict)


@pytest.fixture
def large_feature_df() -> pd.DataFrame:
    """20 providers with 2 fraudulent for stratification tests."""
    rng = np.random.default_rng(42)
    n = 20
    data = {col: rng.random(n) for col in FEATURE_COLS}
    data["Provider"] = [f"P{i:03d}" for i in range(n)]
    data["PotentialFraud"] = [1 if i < 2 else 0 for i in range(n)]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


def test_build_features_shape(feature_df: pd.DataFrame) -> None:
    assert len(feature_df) == 2
    assert "Provider" in feature_df.columns
    assert "PotentialFraud" in feature_df.columns


def test_build_features_has_all_feature_cols(feature_df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in feature_df.columns]
    assert missing == [], f"Missing feature columns: {missing}"


def test_build_features_no_provider_leak(feature_df: pd.DataFrame) -> None:
    assert feature_df["Provider"].nunique() == len(feature_df)


def test_build_features_raises_on_missing_table(tables_dict: dict[str, pd.DataFrame]) -> None:
    incomplete = {k: v for k, v in tables_dict.items() if k != "labels_train"}
    with pytest.raises(ValueError, match="Missing required table"):
        build_features(incomplete)


@pytest.fixture
def claims_flat_tables() -> dict[str, pd.DataFrame]:
    """Mini tabla tipo healthcare_fraud_detection.csv (una fila por reclamación)."""
    df = pd.DataFrame(
        {
            "Provider_ID": ["P001", "P001", "P002", "P002"],
            "Claim_ID": ["C1", "C2", "C3", "C4"],
            "Patient_Age": [40, 41, 50, 51],
            "Patient_Gender": ["Male", "Male", "Female", "Female"],
            "Patient_State": ["NY", "NY", "TX", "TX"],
            "Claim_Amount": np.array([400.0, 500.0, 600.0, 700.0], dtype="float32"),
            "Approved_Amount": np.array([380.0, 490.0, 580.0, 690.0], dtype="float32"),
            "Days_Between_Service_and_Claim": [5, 10, 3, 4],
            "Length_of_Stay": [0, 2, 1, 3],
            "Visit_Type": ["Outpatient", "Inpatient", "Emergency", "Inpatient"],
            "Provider_Specialty": ["Cardiology", "Cardiology", "GP", "GP"],
            "Chronic_Condition_Flag": [1, 0, 1, 1],
            "Is_Fraud": [0, 0, 0, 1],
        }
    )
    return {"claims_flat": df}


def test_build_features_claims_flat_provider_level(
    claims_flat_tables: dict[str, pd.DataFrame],
) -> None:
    out = build_features(claims_flat_tables)
    assert len(out) == 2
    assert set(out["Provider"]) == {"P001", "P002"}
    assert out["PotentialFraud"].tolist() == [0, 1]
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    assert missing == [], f"Missing feature columns: {missing}"


# ---------------------------------------------------------------------------
# split_providers
# ---------------------------------------------------------------------------


def test_split_providers_no_overlap(large_feature_df: pd.DataFrame) -> None:
    train_df, val_df = split_providers(large_feature_df, train_ratio=0.8)
    overlap = set(train_df["Provider"]) & set(val_df["Provider"])
    assert overlap == set()


def test_split_providers_total_rows(large_feature_df: pd.DataFrame) -> None:
    train_df, val_df = split_providers(large_feature_df, train_ratio=0.8)
    assert len(train_df) + len(val_df) == len(large_feature_df)


def test_split_providers_raises_on_missing_target(large_feature_df: pd.DataFrame) -> None:
    df_no_target = large_feature_df.drop(columns=["PotentialFraud"])
    with pytest.raises(ValueError, match="PotentialFraud"):
        split_providers(df_no_target)


# ---------------------------------------------------------------------------
# prepare_train_val
# ---------------------------------------------------------------------------


def test_prepare_train_val_shapes(large_feature_df: pd.DataFrame) -> None:
    train_df, val_df = split_providers(large_feature_df, train_ratio=0.8)
    X_train, X_val, y_train, y_val, _ = prepare_train_val(train_df, val_df)
    assert X_train.shape[1] == len(FEATURE_COLS)
    assert X_val.shape[1] == len(FEATURE_COLS)
    assert len(X_train) + len(X_val) == len(large_feature_df)


def test_prepare_train_val_no_nan(large_feature_df: pd.DataFrame) -> None:
    train_df, val_df = split_providers(large_feature_df, train_ratio=0.8)
    X_train, X_val, _, _, _ = prepare_train_val(train_df, val_df)
    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()


def test_preprocessing_pipeline_no_fit_on_val() -> None:
    rng = np.random.default_rng(0)
    X_train = rng.random((50, len(FEATURE_COLS)))
    X_val = rng.random((10, len(FEATURE_COLS))) * 100  # very different scale

    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)

    train_mean = pipeline.named_steps["scaler"].mean_
    assert train_mean is not None
    # mean_ comes from train data, not from the much-larger val data
    assert not np.allclose(train_mean, X_val.mean(axis=0), rtol=0.5)
