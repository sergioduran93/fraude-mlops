"""Ingeniería de features: de tablas CMS o CSV plano a una fila por proveedor.

El modelo opera a nivel agregado (comportamiento de facturación), no a nivel reclamación
individual, alineado con la etiqueta ``PotentialFraud`` del dataset.
"""

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
    # Agregación principal: señales de volumen, monto, complejidad y perfiles de beneficiarios.
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


_CMS_TABLE_KEYS = frozenset({"inpatient", "outpatient", "beneficiary", "labels_train"})


def _build_features_from_claims_flat(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate consolidated claim-level CSV (`claims_flat`) to provider-level rows.

    Reutiliza los mismos nombres que ``FEATURE_COLS`` del pipeline CMS.
    ``unique_benes`` es un proxy (cardinalidad edad+género+estado); no hay ``BeneID``.
    """
    required = {"Provider_ID", "Claim_ID", "Is_Fraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"claims_flat missing columns: {sorted(missing)}")

    g = df.copy()
    g["Patient_Gender"] = g["Patient_Gender"].astype(str).str.strip()
    g["_patient_key"] = (
        g["Patient_Age"].astype(str)
        + "_"
        + g["Patient_Gender"]
        + "_"
        + g["Patient_State"].astype(str).str.strip()
    )

    if "Visit_Type" in g.columns:
        vt = g["Visit_Type"].astype(str).str.strip().str.lower()
        g["_is_ip"] = (vt == "inpatient").astype("int8")
        g["_is_op"] = vt.isin(["outpatient", "emergency"]).astype("int8")
    else:
        z = np.zeros(len(g), dtype="int8")
        g["_is_ip"] = z
        g["_is_op"] = z

    if {"Claim_Amount", "Approved_Amount"}.issubset(g.columns):
        g["_deduct_proxy"] = (g["Claim_Amount"] - g["Approved_Amount"]).clip(lower=0)
    else:
        g["_deduct_proxy"] = np.nan

    dur_col = (
        "Days_Between_Service_and_Claim" if "Days_Between_Service_and_Claim" in g.columns else None
    )
    stay_col = "Length_of_Stay" if "Length_of_Stay" in g.columns else None
    chronic_col = "Chronic_Condition_Flag" if "Chronic_Condition_Flag" in g.columns else None

    spec: dict[str, tuple[str, str]] = {
        "total_claims": ("Claim_ID", "count"),
        "ip_claims": ("_is_ip", "sum"),
        "op_claims": ("_is_op", "sum"),
        "unique_benes": ("_patient_key", "nunique"),
        "total_reimbursed": ("Claim_Amount", "sum"),
        "mean_reimbursed": ("Claim_Amount", "mean"),
        "max_reimbursed": ("Claim_Amount", "max"),
        "total_deductible": ("_deduct_proxy", "sum"),
        "mean_bene_age": ("Patient_Age", "mean"),
        "PotentialFraud": ("Is_Fraud", "max"),
    }
    if dur_col:
        spec["mean_claim_duration"] = (dur_col, "mean")
    if stay_col:
        spec["mean_hosp_stay"] = (stay_col, "mean")
    else:
        g["_stay_na"] = np.nan
        spec["mean_hosp_stay"] = ("_stay_na", "mean")
    if chronic_col:
        spec["mean_chronic_count"] = (chronic_col, "mean")
    else:
        g["_chronic_na"] = np.nan
        spec["mean_chronic_count"] = ("_chronic_na", "mean")

    if "Provider_Specialty" in g.columns:
        spec["unique_attending"] = ("Provider_Specialty", "nunique")

    agg = g.groupby("Provider_ID", as_index=False).agg(**spec)
    if "unique_attending" not in agg.columns:
        agg["unique_attending"] = np.nan
    if "mean_claim_duration" not in agg.columns:
        agg["mean_claim_duration"] = np.nan

    agg["pct_deceased"] = 0.0
    safe_benes = agg["unique_benes"].replace(0, np.nan)
    agg["claims_per_bene"] = agg["total_claims"] / safe_benes
    agg["reimbursed_per_bene"] = agg["total_reimbursed"] / safe_benes

    agg = agg.rename(columns={"Provider_ID": "Provider"})
    agg["PotentialFraud"] = agg["PotentialFraud"].astype("int8")

    logger.info(
        "claims_flat → provider matrix: %d providers, %d columns",
        len(agg),
        len(agg.columns),
    )
    return agg


def build_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge and aggregate raw tables into a provider-level feature matrix."""
    if "claims_flat" in tables and not _CMS_TABLE_KEYS.issubset(tables.keys()):
        return _build_features_from_claims_flat(tables["claims_flat"])

    for key in _CMS_TABLE_KEYS:
        if key not in tables:
            raise ValueError(f"Missing required table: '{key}'")

    claims = _merge_claims(tables["inpatient"], tables["outpatient"])
    claims_enriched = _enrich_with_beneficiary(claims, tables["beneficiary"])
    provider_features = _aggregate_by_provider(claims_enriched)

    labels = tables["labels_train"][["Provider", "PotentialFraud"]]
    result = provider_features.merge(labels, on="Provider", how="inner")
    logger.info("Feature matrix: %d providers, %d columns", *result.shape)
    return result
