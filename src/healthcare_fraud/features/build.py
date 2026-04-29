"""Feature engineering: aggregate raw tables to provider-level features."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REF_DATE: pd.Timestamp = pd.Timestamp("2010-01-01")

_CHRONIC_COLS: list[str] = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]

_BENE_COLS: list[str] = [
    "BeneID",
    "DOB",
    "DOD",
    "Gender",
    "Race",
    "State",
    "County",
    "NoOfMonths_PartACov",
    "NoOfMonths_PartBCov",
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt",
    "RenalDiseaseIndicator",
] + _CHRONIC_COLS

_COMMON_CLAIM_COLS: list[str] = [
    "BeneID",
    "ClaimID",
    "Provider",
    "ClaimStartDt",
    "ClaimEndDt",
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "AttendingPhysician",
]


def _prepare_inpatient(df: pd.DataFrame) -> pd.DataFrame:
    ip = df.copy()
    ip["claim_type"] = "inpatient"
    if "AdmissionDt" in ip.columns and "DischargeDt" in ip.columns:
        admit = pd.to_datetime(ip["AdmissionDt"], errors="coerce")
        discharge = pd.to_datetime(ip["DischargeDt"], errors="coerce")
        ip["hosp_stay_days"] = (discharge - admit).dt.days.clip(lower=0).astype(float)
    else:
        ip["hosp_stay_days"] = np.nan
    cols = [c for c in _COMMON_CLAIM_COLS if c in ip.columns] + ["claim_type", "hosp_stay_days"]
    return ip[cols]


def _prepare_outpatient(df: pd.DataFrame) -> pd.DataFrame:
    op = df.copy()
    op["claim_type"] = "outpatient"
    op["hosp_stay_days"] = np.nan
    cols = [c for c in _COMMON_CLAIM_COLS if c in op.columns] + ["claim_type", "hosp_stay_days"]
    return op[cols]


def _merge_claims(inpatient: pd.DataFrame, outpatient: pd.DataFrame) -> pd.DataFrame:
    ip_prep = _prepare_inpatient(inpatient)
    op_prep = _prepare_outpatient(outpatient)
    claims = pd.concat([ip_prep, op_prep], ignore_index=True)
    start = pd.to_datetime(claims["ClaimStartDt"], errors="coerce")
    end = pd.to_datetime(claims["ClaimEndDt"], errors="coerce")
    claims["claim_duration_days"] = (end - start).dt.days.clip(lower=0).astype(float)
    logger.info("Claims merged: %d rows", len(claims))
    return claims


def _enrich_with_beneficiary(claims: pd.DataFrame, beneficiary: pd.DataFrame) -> pd.DataFrame:
    bene_cols = [c for c in _BENE_COLS if c in beneficiary.columns]
    enriched = claims.merge(beneficiary[bene_cols], on="BeneID", how="left")
    if "DOB" in enriched.columns:
        dob = pd.to_datetime(enriched["DOB"], errors="coerce")
        enriched["beneficiary_age_years"] = (_REF_DATE - dob).dt.days / 365.25
    else:
        enriched["beneficiary_age_years"] = np.nan
    if "DOD" in enriched.columns:
        enriched["is_deceased"] = enriched["DOD"].notna().astype("int8")
    else:
        enriched["is_deceased"] = 0
    chronic_present = [c for c in _CHRONIC_COLS if c in enriched.columns]
    enriched["chronic_count"] = enriched[chronic_present].sum(axis=1) if chronic_present else 0
    return enriched


def _aggregate_by_provider(claims_enriched: pd.DataFrame) -> pd.DataFrame:
    df = claims_enriched.copy()
    df["is_inpatient"] = (df["claim_type"] == "inpatient").astype("int8")
    df["is_outpatient"] = (df["claim_type"] == "outpatient").astype("int8")

    agg_spec: dict = {
        "total_claims": ("ClaimID", "count"),
        "ip_claims": ("is_inpatient", "sum"),
        "op_claims": ("is_outpatient", "sum"),
        "unique_benes": ("BeneID", "nunique"),
        "total_reimbursed": ("InscClaimAmtReimbursed", "sum"),
        "mean_reimbursed": ("InscClaimAmtReimbursed", "mean"),
        "max_reimbursed": ("InscClaimAmtReimbursed", "max"),
        "mean_claim_duration": ("claim_duration_days", "mean"),
        "mean_hosp_stay": ("hosp_stay_days", "mean"),
        "mean_bene_age": ("beneficiary_age_years", "mean"),
        "pct_deceased": ("is_deceased", "mean"),
        "mean_chronic_count": ("chronic_count", "mean"),
    }
    if "DeductibleAmtPaid" in df.columns:
        agg_spec["total_deductible"] = ("DeductibleAmtPaid", "sum")
    if "AttendingPhysician" in df.columns:
        agg_spec["unique_attending"] = ("AttendingPhysician", "nunique")

    agg = df.groupby("Provider").agg(**agg_spec).reset_index()

    if "total_deductible" not in agg.columns:
        agg["total_deductible"] = np.nan
    if "unique_attending" not in agg.columns:
        agg["unique_attending"] = np.nan

    safe_benes = agg["unique_benes"].replace(0, np.nan)
    agg["claims_per_bene"] = agg["total_claims"] / safe_benes
    agg["reimbursed_per_bene"] = agg["total_reimbursed"] / safe_benes

    logger.info("Provider aggregation: %d providers, %d features", len(agg), len(agg.columns) - 1)
    return agg


def build_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge and aggregate raw tables into a provider-level feature matrix."""
    for key in ("inpatient", "outpatient", "beneficiary", "labels_train"):
        if key not in tables:
            raise ValueError(f"Missing required table: '{key}'")

    claims = _merge_claims(tables["inpatient"], tables["outpatient"])
    claims_enriched = _enrich_with_beneficiary(claims, tables["beneficiary"])
    provider_features = _aggregate_by_provider(claims_enriched)

    labels = tables["labels_train"][["Provider", "PotentialFraud"]]
    result = provider_features.merge(labels, on="Provider", how="inner")
    logger.info("Feature matrix: %d providers, %d columns", *result.shape)
    return result
