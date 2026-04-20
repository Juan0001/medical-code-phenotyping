"""
End-to-end diabetic cohort identification from synthetic claims data.
Serves as the tested reference implementation for the companion notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

# -------------------------------------------------------------------
# 1. Synthetic data generation
# -------------------------------------------------------------------
ICD_DM_PREFIXES = ("E10", "E11", "E12", "E13")

# Illustrative antidiabetic RxNorm concepts (ingredient-level RxCUIs)
ANTIDM_RXCUIS = {
    6809: "metformin",
    5856: "insulin",
    593411: "sitagliptin",
    857974: "glipizide",
    1373458: "empagliflozin",
    1551291: "liraglutide",
    860975: "dulaglutide",
    200370: "glyburide",
}
NON_DM_RXCUIS = {
    617312: "atorvastatin",
    617314: "simvastatin",
    197319: "amlodipine",
    310965: "lisinopril",
    1044427: "gabapentin",
    866924: "metoprolol",
}
DM_ICDS = ["E11.9", "E11.65", "E11.22", "E11.9", "E10.9", "E13.9"]
NON_DM_ICDS = ["I10", "M25.5", "R51", "J45.9", "Z00.00", "N39.0", "I25.10", "M54.5"]
A1C_LOINCS = {"4548-4", "17856-6", "41995-2"}


def generate_synthetic_data(n_patients: int = 5000, seed: int = 42):
    """Generate synthetic patient, diagnosis, pharmacy, lab, and procedure tables."""
    rng = np.random.default_rng(seed)

    patients = pd.DataFrame(
        {
            "patient_id": [f"P{i:06d}" for i in range(n_patients)],
            "age": rng.integers(20, 85, n_patients),
            "sex": rng.choice(["M", "F"], n_patients),
        }
    )
    truth = rng.random(n_patients) < 0.22
    patients_truth = patients[["patient_id"]].copy()
    patients_truth["is_diabetic_truth"] = truth

    # Diagnosis claims -----------------------------------------------
    base = rng.poisson(5, n_patients)
    extra = rng.poisson(4, n_patients) * truth.astype(int)
    n_claims = base + extra
    total = n_claims.sum()
    pid_idx = np.repeat(np.arange(n_patients), n_claims)
    is_dm_claim = (rng.random(total) < 0.55) & truth[pid_idx]
    codes = np.where(
        is_dm_claim,
        rng.choice(DM_ICDS, total),
        rng.choice(NON_DM_ICDS, total),
    )
    # Low-rate false-positive coding noise for non-diabetics
    noise_mask = (rng.random(total) < 0.015) & ~truth[pid_idx]
    codes = np.where(noise_mask, "E11.9", codes)
    claim_types = rng.choice(["outpatient", "inpatient"], total, p=[0.88, 0.12])
    service_dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        rng.integers(0, 365, total), unit="D"
    )
    diagnosis_claims = pd.DataFrame(
        {
            "patient_id": patients["patient_id"].values[pid_idx],
            "claim_id": [f"DX{i:09d}" for i in range(total)],
            "service_date": service_dates,
            "claim_type": claim_types,
            "icd_code": codes,
        }
    )

    # Pharmacy claims ------------------------------------------------
    base_rx = rng.poisson(1, n_patients)
    extra_rx = rng.poisson(4, n_patients) * truth.astype(int)
    n_rx = base_rx + extra_rx
    total_rx = n_rx.sum()
    pid_idx_rx = np.repeat(np.arange(n_patients), n_rx)
    is_antidm = (rng.random(total_rx) < 0.70) & truth[pid_idx_rx]
    antidm_codes = np.array(list(ANTIDM_RXCUIS.keys()))
    non_dm_codes = np.array(list(NON_DM_RXCUIS.keys()))
    rxcui = np.where(
        is_antidm,
        rng.choice(antidm_codes, total_rx),
        rng.choice(non_dm_codes, total_rx),
    )
    name_map = {**ANTIDM_RXCUIS, **NON_DM_RXCUIS}
    pharmacy_claims = pd.DataFrame(
        {
            "patient_id": patients["patient_id"].values[pid_idx_rx],
            "claim_id": [f"RX{i:09d}" for i in range(total_rx)],
            "service_date": pd.to_datetime("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, total_rx), unit="D"),
            "rxnorm_code": rxcui,
            "drug_name": [name_map[c] for c in rxcui],
        }
    )

    # Lab results (A1c) ---------------------------------------------
    has_a1c = ((rng.random(n_patients) < 0.85) & truth) | (rng.random(n_patients) < 0.12)
    n_a1c = rng.integers(1, 4, n_patients) * has_a1c.astype(int)
    total_a1c = int(n_a1c.sum())
    pid_idx_lab = np.repeat(np.arange(n_patients), n_a1c)
    a1c_values = np.where(
        truth[pid_idx_lab],
        rng.normal(8.0, 1.5, total_a1c),
        rng.normal(5.5, 0.4, total_a1c),
    ).clip(4.0, 14.0).round(1)
    lab_results = pd.DataFrame(
        {
            "patient_id": patients["patient_id"].values[pid_idx_lab],
            "service_date": pd.to_datetime("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, total_a1c), unit="D"),
            "loinc_code": rng.choice(list(A1C_LOINCS), total_a1c),
            "test_name": "Hemoglobin A1c",
            "value": a1c_values,
            "unit": "%",
        }
    )

    # Procedure claims (include CPT 83036 for every A1c lab) --------
    a1c_proc = lab_results[["patient_id", "service_date"]].copy()
    a1c_proc["cpt_code"] = "83036"
    a1c_proc["claim_id"] = [f"PR{i:09d}" for i in range(len(a1c_proc))]
    # Unrelated procedures
    n_extra = n_patients * 3
    extra_proc = pd.DataFrame(
        {
            "patient_id": rng.choice(patients["patient_id"], n_extra),
            "claim_id": [
                f"PR{i:09d}" for i in range(len(a1c_proc), len(a1c_proc) + n_extra)
            ],
            "service_date": pd.to_datetime("2025-01-01")
            + pd.to_timedelta(rng.integers(0, 365, n_extra), unit="D"),
            "cpt_code": rng.choice(
                ["99213", "99214", "80053", "85025", "93000"], n_extra
            ),
        }
    )
    procedure_claims = pd.concat(
        [a1c_proc, extra_proc], ignore_index=True
    )[["patient_id", "claim_id", "service_date", "cpt_code"]]

    return patients, patients_truth, diagnosis_claims, pharmacy_claims, lab_results, procedure_claims


# -------------------------------------------------------------------
# 2. Schema normalization
# -------------------------------------------------------------------
def normalize_claims(df: pd.DataFrame, date_col: str = "service_date") -> pd.DataFrame:
    """Normalize column dtypes for downstream processing."""
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    for c in ("icd_code", "cpt_code", "loinc_code", "claim_type"):
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


# -------------------------------------------------------------------
# 3. Phenotype logic
# -------------------------------------------------------------------
def filter_diabetes_diagnoses(df_dx: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose ICD-10-CM code starts with an E10–E13 prefix."""
    code = df_dx["icd_code"].astype(str)
    mask = code.str.startswith(ICD_DM_PREFIXES)
    return df_dx[mask]


def patients_with_outpatient_dx(df_dx_dm: pd.DataFrame, min_distinct_dates: int = 2) -> set:
    outp = df_dx_dm[df_dx_dm["claim_type"] == "outpatient"]
    distinct = outp.groupby("patient_id", observed=True)["service_date"].nunique()
    return set(distinct[distinct >= min_distinct_dates].index)


def patients_with_inpatient_dx(df_dx_dm: pd.DataFrame) -> set:
    inp = df_dx_dm[df_dx_dm["claim_type"] == "inpatient"]
    return set(inp["patient_id"].unique())


def patients_with_antidm_rx(df_rx: pd.DataFrame) -> set:
    mask = df_rx["rxnorm_code"].isin(ANTIDM_RXCUIS.keys())
    return set(df_rx[mask]["patient_id"].unique())


# -------------------------------------------------------------------
# 4. Cohort construction
# -------------------------------------------------------------------
def build_cohort(patients: pd.DataFrame, df_dx: pd.DataFrame, df_rx: pd.DataFrame) -> pd.DataFrame:
    dm_dx = filter_diabetes_diagnoses(df_dx)
    out = patients_with_outpatient_dx(dm_dx)
    inp = patients_with_inpatient_dx(dm_dx)
    rx = patients_with_antidm_rx(df_rx)

    cohort = patients.copy()
    cohort["has_outpatient_dx"] = cohort["patient_id"].isin(out)
    cohort["has_inpatient_dx"] = cohort["patient_id"].isin(inp)
    cohort["has_antidm_rx"] = cohort["patient_id"].isin(rx)
    cohort["is_diabetic"] = cohort[
        ["has_outpatient_dx", "has_inpatient_dx", "has_antidm_rx"]
    ].any(axis=1)
    return cohort


# -------------------------------------------------------------------
# 5. Lab enrichment
# -------------------------------------------------------------------
def enrich_with_a1c(cohort: pd.DataFrame, df_lab: pd.DataFrame) -> pd.DataFrame:
    a1c = df_lab[df_lab["loinc_code"].isin(A1C_LOINCS)].copy()
    if a1c.empty:
        for col in ("latest_a1c", "mean_a1c", "max_a1c", "n_a1c", "latest_a1c_date"):
            cohort[col] = np.nan
        cohort["poor_control"] = False
        return cohort
    agg = a1c.groupby("patient_id").agg(
        mean_a1c=("value", "mean"),
        max_a1c=("value", "max"),
        n_a1c=("value", "size"),
        latest_a1c_date=("service_date", "max"),
    )
    a1c_sorted = a1c.sort_values(["patient_id", "service_date"])
    latest = a1c_sorted.groupby("patient_id").tail(1).set_index("patient_id")["value"].rename("latest_a1c")
    agg = agg.join(latest)
    out = cohort.merge(agg, on="patient_id", how="left")
    out["poor_control"] = out["latest_a1c"].fillna(0) > 8.0
    return out


# -------------------------------------------------------------------
# 6. Procedure validation
# -------------------------------------------------------------------
def add_a1c_testing_flag(cohort: pd.DataFrame, df_proc: pd.DataFrame) -> pd.DataFrame:
    testers = set(df_proc[df_proc["cpt_code"] == "83036"]["patient_id"].unique())
    out = cohort.copy()
    out["has_a1c_testing"] = out["patient_id"].isin(testers)
    return out


# -------------------------------------------------------------------
# 7. Feature engineering
# -------------------------------------------------------------------
def engineer_features(cohort: pd.DataFrame, df_rx: pd.DataFrame) -> pd.DataFrame:
    dm_rx = df_rx[df_rx["rxnorm_code"].isin(ANTIDM_RXCUIS.keys())]
    n_agents = dm_rx.groupby("patient_id")["rxnorm_code"].nunique()
    insulin = set(dm_rx[dm_rx["rxnorm_code"] == 5856]["patient_id"].unique())
    metformin = set(dm_rx[dm_rx["rxnorm_code"] == 6809]["patient_id"].unique())
    glp1 = set(dm_rx[dm_rx["rxnorm_code"].isin({1551291, 860975})]["patient_id"].unique())
    sglt2 = set(dm_rx[dm_rx["rxnorm_code"] == 1373458]["patient_id"].unique())

    out = cohort.copy()
    out["n_antidm_agents"] = out["patient_id"].map(n_agents).fillna(0).astype(int)
    out["on_insulin"] = out["patient_id"].isin(insulin)
    out["on_metformin"] = out["patient_id"].isin(metformin)
    out["on_glp1"] = out["patient_id"].isin(glp1)
    out["on_sglt2"] = out["patient_id"].isin(sglt2)
    return out


# -------------------------------------------------------------------
# 8. Scalable variant (chunked Parquet)
# -------------------------------------------------------------------
def streaming_diabetes_flag(parquet_paths, chunk_size: int = 1_000_000):
    """Illustrative chunked processing for very large claims feeds."""
    accumulators = {"outpatient_dates": {}, "inpatient": set(), "rx": set()}
    for path in parquet_paths:
        parquet_iter = pd.read_parquet(path)  # single shard per path
        # In real pipelines you'd use pyarrow.dataset with row-group streaming
        for start in range(0, len(parquet_iter), chunk_size):
            chunk = parquet_iter.iloc[start : start + chunk_size]
            dm = filter_diabetes_diagnoses(chunk)
            outp = dm[dm["claim_type"] == "outpatient"]
            for pid, s in outp.groupby("patient_id")["service_date"]:
                accumulators["outpatient_dates"].setdefault(pid, set()).update(s)
            accumulators["inpatient"].update(dm[dm["claim_type"] == "inpatient"]["patient_id"].unique())
    diabetic_pids = set(accumulators["inpatient"])
    for pid, dates in accumulators["outpatient_dates"].items():
        if len(dates) >= 2:
            diabetic_pids.add(pid)
    return diabetic_pids


# -------------------------------------------------------------------
# 9. Main demo
# -------------------------------------------------------------------
def main():
    patients, truth, dx, rx, labs, procs = generate_synthetic_data(n_patients=5000)
    dx, rx, labs, procs = map(normalize_claims, (dx, rx, labs, procs))

    cohort = build_cohort(patients, dx, rx)
    cohort = enrich_with_a1c(cohort, labs)
    cohort = add_a1c_testing_flag(cohort, procs)
    cohort = engineer_features(cohort, rx)

    # Evaluate against simulated ground truth
    eval_df = cohort.merge(truth, on="patient_id")
    tp = ((eval_df.is_diabetic) & (eval_df.is_diabetic_truth)).sum()
    fp = ((eval_df.is_diabetic) & (~eval_df.is_diabetic_truth)).sum()
    fn = ((~eval_df.is_diabetic) & (eval_df.is_diabetic_truth)).sum()
    tn = ((~eval_df.is_diabetic) & (~eval_df.is_diabetic_truth)).sum()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0

    print(f"Patients: {len(cohort):,}")
    print(f"Diabetic cohort: {cohort.is_diabetic.sum():,}")
    print(f"Sensitivity={sens:.3f}  Specificity={spec:.3f}  PPV={ppv:.3f}")
    print("\nSample cohort rows:")
    print(cohort[cohort.is_diabetic].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
