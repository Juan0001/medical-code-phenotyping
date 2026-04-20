"""
Diabetes cohort identification using the MIMIC-IV schema.

This module is designed to work against:
  * The open PhysioNet demo ("MIMIC-IV Clinical Database Demo v2.2", 100 patients)
  * The full credentialed MIMIC-IV 2.2 / 3.x (hundreds of thousands of patients)

If the real CSVs are not present on disk, a MIMIC-IV-schema-faithful synthetic
fixture is generated so the pipeline can still be demonstrated.

Phenotype (adapted for a largely inpatient dataset):
  (A) >= 2 distinct hospital admissions with a diabetes ICD code (E10-E13 / 250.*)
  (B) >= 1 admission whose PRIMARY (seq_num == 1) diagnosis is diabetes
  (C) >= 1 prescription for an antidiabetic agent
Enrichment:
  Hemoglobin A1c via labevents + d_labitems (LOINC 4548-4 / 17856-6)
"""
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 0. Constants
# ---------------------------------------------------------------
ICD10_DM_PREFIXES = ("E10", "E11", "E12", "E13")
ICD9_DM_PREFIX = "250"

A1C_LOINCS = {"4548-4", "17856-6", "41995-2"}
A1C_LABEL_KEYWORDS = ("hemoglobin a1c", "hba1c", "hgb a1c", "glycated hemoglobin")

# MIMIC-IV stores drug names as free text. We filter by lowercase substring match
# against a curated list that covers the major antidiabetic classes.
ANTIDM_DRUG_TOKENS = [
    # Biguanide
    "metformin",
    # Insulins (brand + generic)
    "insulin", "humalog", "novolog", "novolin", "humulin",
    "lantus", "levemir", "tresiba", "apidra", "toujeo", "basaglar",
    # Sulfonylureas
    "glipizide", "glyburide", "glimepiride", "glibenclamide",
    # DPP-4
    "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
    "januvia", "onglyza", "tradjenta",
    # SGLT-2
    "empagliflozin", "canagliflozin", "dapagliflozin", "ertugliflozin",
    "jardiance", "invokana", "farxiga", "steglatro",
    # GLP-1
    "liraglutide", "dulaglutide", "semaglutide", "exenatide", "lixisenatide",
    "victoza", "trulicity", "ozempic", "rybelsus", "byetta", "bydureon",
    # TZDs
    "pioglitazone", "rosiglitazone", "actos", "avandia",
    # Meglitinides
    "repaglinide", "nateglinide",
]

# ---------------------------------------------------------------
# 1. Loader
# ---------------------------------------------------------------
def load_mimic(
    mimic_root: str | None = None,
    synthetic_n: int = 300,
    seed: int = 42,
):
    """
    Load MIMIC-IV tables from disk, or generate a schema-faithful synthetic
    fixture if the files are not present.

    Returns
    -------
    tables : dict[str, pd.DataFrame]
        Keys: patients, admissions, diagnoses_icd, prescriptions,
              labevents, d_labitems
    source : str
        "mimic" if real files were used, "synthetic" otherwise
    """
    if mimic_root is None:
        mimic_root = os.environ.get("MIMIC_ROOT", "./mimic-iv-demo")
    root = Path(mimic_root)

    # Real CSVs live under <root>/hosp/<table>.csv.gz for the PhysioNet demo
    hosp = root / "hosp"
    expected = ["patients", "admissions", "diagnoses_icd",
                "prescriptions", "labevents", "d_labitems"]

    def _read_table(name):
        for suffix in (".csv.gz", ".csv"):
            p = hosp / f"{name}{suffix}"
            if p.exists():
                return pd.read_csv(p, low_memory=False)
        return None

    if hosp.exists():
        tables = {n: _read_table(n) for n in expected}
        if all(t is not None for t in tables.values()):
            return tables, "mimic"

    return _generate_mimic_fixture(n_patients=synthetic_n, seed=seed), "synthetic"


# ---------------------------------------------------------------
# 2. Synthetic MIMIC-IV-shaped fixture
# ---------------------------------------------------------------
def _generate_mimic_fixture(n_patients: int = 300, seed: int = 42):
    """Return a dict of DataFrames mirroring MIMIC-IV column conventions."""
    rng = np.random.default_rng(seed)

    # ---- patients ------------------------------------------------
    subject_ids = rng.integers(10_000_000, 19_999_999, n_patients)
    subject_ids = np.unique(subject_ids)[:n_patients]
    # Ensure exactly n_patients unique ids
    while len(subject_ids) < n_patients:
        extra = rng.integers(10_000_000, 19_999_999, n_patients - len(subject_ids))
        subject_ids = np.unique(np.concatenate([subject_ids, extra]))
    subject_ids = subject_ids[:n_patients]

    truth = rng.random(n_patients) < 0.25  # ICU populations have higher DM prevalence
    patients = pd.DataFrame({
        "subject_id": subject_ids,
        "gender": rng.choice(["M", "F"], n_patients),
        "anchor_age": rng.integers(18, 90, n_patients),
        "anchor_year": rng.integers(2008, 2019, n_patients),
        "anchor_year_group": rng.choice(
            ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"],
            n_patients,
        ),
        "dod": pd.NaT,
    })

    # ---- admissions ----------------------------------------------
    n_adm_per_pt = rng.poisson(1.2, n_patients) + 1 + rng.poisson(1.0, n_patients) * truth.astype(int)
    pid_idx = np.repeat(np.arange(n_patients), n_adm_per_pt)
    n_adm = int(n_adm_per_pt.sum())
    admit_days = rng.integers(0, 3650, n_adm)
    admittime = pd.to_datetime("2110-01-01") + pd.to_timedelta(admit_days, unit="D")
    los_hours = rng.integers(24, 24 * 14, n_adm)
    dischtime = admittime + pd.to_timedelta(los_hours, unit="h")

    admissions = pd.DataFrame({
        "subject_id": patients["subject_id"].values[pid_idx],
        "hadm_id": rng.integers(20_000_000, 29_999_999, n_adm),
        "admittime": admittime,
        "dischtime": dischtime,
        "deathtime": pd.NaT,
        "admission_type": rng.choice(
            ["ELECTIVE", "EW EMER.", "URGENT", "OBSERVATION ADMIT", "DIRECT EMER."],
            n_adm,
        ),
        "admission_location": "EMERGENCY ROOM",
        "discharge_location": "HOME",
        "insurance": rng.choice(["Medicare", "Medicaid", "Private", "Other"], n_adm),
        "language": "ENGLISH",
        "marital_status": rng.choice(["MARRIED", "SINGLE", "DIVORCED", "WIDOWED"], n_adm),
        "ethnicity": "UNKNOWN",
        "edregtime": pd.NaT,
        "edouttime": pd.NaT,
        "hospital_expire_flag": 0,
    })

    # ---- diagnoses_icd -------------------------------------------
    # Each admission gets 5-15 diagnosis lines with seq_num 1..N
    n_dx_per_adm = rng.integers(5, 16, n_adm)
    total_dx = int(n_dx_per_adm.sum())
    adm_idx_dx = np.repeat(np.arange(n_adm), n_dx_per_adm)
    seq_num = np.concatenate([np.arange(1, k + 1) for k in n_dx_per_adm])

    # Each admission belonging to a true-diabetic patient has a higher chance of
    # a diabetes code, and a moderate chance of it being the primary diagnosis.
    adm_patient_truth = truth[pid_idx]
    is_primary = seq_num == 1
    p_dm_primary = np.where(adm_patient_truth[adm_idx_dx] & is_primary, 0.35, 0.02)
    p_dm_secondary = np.where(adm_patient_truth[adm_idx_dx] & ~is_primary, 0.25, 0.01)
    p_dm = np.where(is_primary, p_dm_primary, p_dm_secondary)
    is_dm_dx = rng.random(total_dx) < p_dm

    icd10_dm = ["E11.9", "E11.65", "E11.22", "E10.9", "E13.9", "E11.40"]
    icd9_dm = ["25000", "25002", "25040", "25060", "25080"]
    icd10_other = ["I10", "I25.10", "N17.9", "J96.01", "Z79.4", "E78.5", "I50.9", "K21.9"]
    icd9_other = ["4019", "41401", "5849", "2724", "V5867", "42831", "5990"]

    version = rng.choice([9, 10], total_dx, p=[0.25, 0.75])
    codes = np.empty(total_dx, dtype=object)
    mask_dm10 = is_dm_dx & (version == 10)
    mask_dm9 = is_dm_dx & (version == 9)
    mask_o10 = ~is_dm_dx & (version == 10)
    mask_o9 = ~is_dm_dx & (version == 9)
    codes[mask_dm10] = rng.choice(icd10_dm, mask_dm10.sum())
    codes[mask_dm9] = rng.choice(icd9_dm, mask_dm9.sum())
    codes[mask_o10] = rng.choice(icd10_other, mask_o10.sum())
    codes[mask_o9] = rng.choice(icd9_other, mask_o9.sum())

    diagnoses_icd = pd.DataFrame({
        "subject_id": admissions["subject_id"].values[adm_idx_dx],
        "hadm_id": admissions["hadm_id"].values[adm_idx_dx],
        "seq_num": seq_num,
        "icd_code": codes,
        "icd_version": version,
    })

    # ---- prescriptions -------------------------------------------
    n_rx_per_adm = rng.integers(3, 12, n_adm)
    total_rx = int(n_rx_per_adm.sum())
    adm_idx_rx = np.repeat(np.arange(n_adm), n_rx_per_adm)
    adm_pt_truth = truth[pid_idx[adm_idx_rx]]
    is_antidm = rng.random(total_rx) < np.where(adm_pt_truth, 0.35, 0.01)

    antidm_drugs = [
        "Metformin", "Metformin HCl", "MetFORMIN",
        "Insulin Lispro", "Insulin Glargine", "Insulin Aspart",
        "HumaLOG", "NovoLOG", "Lantus",
        "Glipizide", "Glyburide", "Glimepiride",
        "Sitagliptin", "Linagliptin",
        "Empagliflozin", "Dapagliflozin", "Canagliflozin",
        "Liraglutide", "Dulaglutide", "Semaglutide",
    ]
    other_drugs = [
        "Aspirin", "Atorvastatin", "Simvastatin", "Lisinopril", "Amlodipine",
        "Furosemide", "Metoprolol Tartrate", "Pantoprazole", "Heparin",
        "Acetaminophen", "Ondansetron", "Levothyroxine", "Warfarin",
    ]
    drug = np.where(
        is_antidm,
        rng.choice(antidm_drugs, total_rx),
        rng.choice(other_drugs, total_rx),
    )
    starttime = admissions["admittime"].values[adm_idx_rx] + pd.to_timedelta(
        rng.integers(0, 72, total_rx), unit="h"
    )
    prescriptions = pd.DataFrame({
        "subject_id": admissions["subject_id"].values[adm_idx_rx],
        "hadm_id": admissions["hadm_id"].values[adm_idx_rx],
        "pharmacy_id": rng.integers(50_000_000, 59_999_999, total_rx),
        "starttime": starttime,
        "stoptime": starttime + pd.to_timedelta(rng.integers(1, 7, total_rx), unit="D"),
        "drug_type": "MAIN",
        "drug": drug,
        "gsn": rng.integers(1000, 99999, total_rx).astype(str),
        "ndc": rng.integers(10_000_000, 99_999_999, total_rx).astype(str),
        "prod_strength": "",
        "dose_val_rx": rng.integers(1, 500, total_rx).astype(str),
        "dose_unit_rx": "mg",
        "route": "PO",
    })

    # ---- d_labitems ----------------------------------------------
    d_labitems = pd.DataFrame([
        {"itemid": 50852, "label": "% Hemoglobin A1c",             "fluid": "Blood", "category": "Chemistry", "loinc_code": "4548-4"},
        {"itemid": 51277, "label": "Hemoglobin A1c",                "fluid": "Blood", "category": "Hematology", "loinc_code": "17856-6"},
        {"itemid": 50902, "label": "Chloride",                      "fluid": "Blood", "category": "Chemistry", "loinc_code": "2075-0"},
        {"itemid": 50931, "label": "Glucose",                       "fluid": "Blood", "category": "Chemistry", "loinc_code": "2345-7"},
        {"itemid": 51006, "label": "Urea Nitrogen",                 "fluid": "Blood", "category": "Chemistry", "loinc_code": "3094-0"},
        {"itemid": 50971, "label": "Potassium",                     "fluid": "Blood", "category": "Chemistry", "loinc_code": "2823-3"},
    ])
    a1c_itemids = [50852, 51277]
    other_itemids = [50902, 50931, 51006, 50971]

    # ---- labevents -----------------------------------------------
    # Generate A1c and a few other labs per admission
    n_lab_per_adm = rng.integers(0, 4, n_adm)
    total_lab_a1c = int(n_lab_per_adm.sum())
    adm_idx_lab = np.repeat(np.arange(n_adm), n_lab_per_adm)

    # Only diabetics reliably have A1c drawn
    pt_truth_lab = truth[pid_idx[adm_idx_lab]]
    has_a1c = (rng.random(total_lab_a1c) < np.where(pt_truth_lab, 0.9, 0.15))
    a1c_mask = has_a1c
    a1c_values = np.where(
        pt_truth_lab,
        rng.normal(8.2, 1.8, total_lab_a1c),
        rng.normal(5.4, 0.5, total_lab_a1c),
    ).clip(4.0, 15.0).round(1)

    labevents_a1c = pd.DataFrame({
        "labevent_id": np.arange(total_lab_a1c),
        "subject_id": admissions["subject_id"].values[adm_idx_lab],
        "hadm_id": admissions["hadm_id"].values[adm_idx_lab],
        "specimen_id": rng.integers(1, 10**9, total_lab_a1c),
        "itemid": rng.choice(a1c_itemids, total_lab_a1c),
        "charttime": admissions["admittime"].values[adm_idx_lab]
            + pd.to_timedelta(rng.integers(0, 48, total_lab_a1c), unit="h"),
        "storetime": pd.NaT,
        "value": a1c_values.astype(str),
        "valuenum": a1c_values,
        "valueuom": "%",
        "ref_range_lower": 4.0,
        "ref_range_upper": 5.6,
        "flag": np.where(a1c_values > 6.4, "abnormal", ""),
        "priority": "ROUTINE",
        "comments": "",
    })
    labevents_a1c = labevents_a1c[a1c_mask]

    # Add some unrelated labs so A1c isn't the only thing in labevents
    n_other_lab = n_adm * 5
    idx = rng.integers(0, n_adm, n_other_lab)
    lab_other = pd.DataFrame({
        "labevent_id": np.arange(total_lab_a1c, total_lab_a1c + n_other_lab),
        "subject_id": admissions["subject_id"].values[idx],
        "hadm_id": admissions["hadm_id"].values[idx],
        "specimen_id": rng.integers(1, 10**9, n_other_lab),
        "itemid": rng.choice(other_itemids, n_other_lab),
        "charttime": admissions["admittime"].values[idx]
            + pd.to_timedelta(rng.integers(0, 72, n_other_lab), unit="h"),
        "storetime": pd.NaT,
        "value": rng.integers(50, 200, n_other_lab).astype(str),
        "valuenum": rng.integers(50, 200, n_other_lab).astype(float),
        "valueuom": "mg/dL",
        "ref_range_lower": 70.0, "ref_range_upper": 110.0,
        "flag": "", "priority": "ROUTINE", "comments": "",
    })
    labevents = pd.concat([labevents_a1c, lab_other], ignore_index=True)

    return {
        "patients": patients,
        "admissions": admissions,
        "diagnoses_icd": diagnoses_icd,
        "prescriptions": prescriptions,
        "labevents": labevents,
        "d_labitems": d_labitems,
        "_truth": pd.DataFrame({"subject_id": subject_ids, "is_diabetic_truth": truth}),
    }


# ---------------------------------------------------------------
# 3. Phenotype primitives
# ---------------------------------------------------------------
def is_diabetes_icd(code: pd.Series, version: pd.Series) -> pd.Series:
    """Vectorized test: True if the ICD code is a diabetes code."""
    code = code.astype(str).str.replace(".", "", regex=False).str.upper()
    v10 = (version == 10) & code.str.startswith(ICD10_DM_PREFIXES)
    v9 = (version == 9) & code.str.startswith(ICD9_DM_PREFIX)
    return v10 | v9


def admissions_with_dm(diagnoses_icd: pd.DataFrame, primary_only: bool = False) -> pd.DataFrame:
    """Return hadm_id rows whose diagnoses include a diabetes code."""
    dx = diagnoses_icd
    if primary_only:
        dx = dx[dx["seq_num"] == 1]
    dm = dx[is_diabetes_icd(dx["icd_code"], dx["icd_version"])]
    return dm[["subject_id", "hadm_id"]].drop_duplicates()


def patients_with_multi_admission_dm(diagnoses_icd: pd.DataFrame, min_admits: int = 2) -> set:
    dm = admissions_with_dm(diagnoses_icd)
    counts = dm.groupby("subject_id")["hadm_id"].nunique()
    return set(counts[counts >= min_admits].index)


def patients_with_primary_dm(diagnoses_icd: pd.DataFrame) -> set:
    return set(admissions_with_dm(diagnoses_icd, primary_only=True)["subject_id"].unique())


def patients_with_antidm_rx(prescriptions: pd.DataFrame) -> set:
    drug = prescriptions["drug"].astype(str).str.lower()
    mask = pd.Series(False, index=drug.index)
    for tok in ANTIDM_DRUG_TOKENS:
        mask |= drug.str.contains(tok, na=False)
    return set(prescriptions.loc[mask, "subject_id"].unique())


# ---------------------------------------------------------------
# 4. Cohort
# ---------------------------------------------------------------
def build_cohort(tables: dict) -> pd.DataFrame:
    patients = tables["patients"]
    dx = tables["diagnoses_icd"]
    rx = tables["prescriptions"]

    multi = patients_with_multi_admission_dm(dx, min_admits=2)
    primary = patients_with_primary_dm(dx)
    rx_pos = patients_with_antidm_rx(rx)

    cohort = patients[["subject_id", "gender", "anchor_age"]].copy()
    cohort["has_multi_admit_dm"] = cohort["subject_id"].isin(multi)
    cohort["has_primary_dm"] = cohort["subject_id"].isin(primary)
    cohort["has_antidm_rx"] = cohort["subject_id"].isin(rx_pos)
    cohort["is_diabetic"] = cohort[
        ["has_multi_admit_dm", "has_primary_dm", "has_antidm_rx"]
    ].any(axis=1)
    return cohort


# ---------------------------------------------------------------
# 5. A1c enrichment using LOINC-aware lab joins
# ---------------------------------------------------------------
def enrich_with_a1c(cohort: pd.DataFrame, tables: dict) -> pd.DataFrame:
    d_lab = tables["d_labitems"]
    lab = tables["labevents"]

    # Prefer LOINC match; fall back to label keyword match (older MIMIC builds
    # that lack loinc_code in d_labitems)
    a1c_items = set()
    if "loinc_code" in d_lab.columns:
        a1c_items |= set(
            d_lab.loc[d_lab["loinc_code"].isin(A1C_LOINCS), "itemid"].unique()
        )
    label = d_lab["label"].astype(str).str.lower()
    for kw in A1C_LABEL_KEYWORDS:
        a1c_items |= set(d_lab.loc[label.str.contains(kw, na=False), "itemid"].unique())

    a1c = lab[lab["itemid"].isin(a1c_items) & lab["valuenum"].notna()].copy()

    if a1c.empty:
        for c in ("latest_a1c", "mean_a1c", "max_a1c", "n_a1c", "latest_a1c_date"):
            cohort[c] = np.nan
        cohort["poor_control"] = False
        return cohort

    a1c["charttime"] = pd.to_datetime(a1c["charttime"])
    agg = a1c.groupby("subject_id").agg(
        mean_a1c=("valuenum", "mean"),
        max_a1c=("valuenum", "max"),
        n_a1c=("valuenum", "size"),
        latest_a1c_date=("charttime", "max"),
    )
    a1c_sorted = a1c.sort_values(["subject_id", "charttime"])
    latest = (
        a1c_sorted.groupby("subject_id").tail(1)
        .set_index("subject_id")["valuenum"].rename("latest_a1c")
    )
    agg = agg.join(latest)
    out = cohort.merge(agg, on="subject_id", how="left")
    out["poor_control"] = out["latest_a1c"].fillna(0) > 8.0
    return out


# ---------------------------------------------------------------
# 6. Medication-class features
# ---------------------------------------------------------------
def engineer_features(cohort: pd.DataFrame, prescriptions: pd.DataFrame) -> pd.DataFrame:
    drug = prescriptions["drug"].astype(str).str.lower()

    def _patients_on(tokens):
        mask = pd.Series(False, index=drug.index)
        for t in tokens:
            mask |= drug.str.contains(t, na=False)
        return set(prescriptions.loc[mask, "subject_id"].unique())

    insulin = _patients_on(["insulin", "humalog", "novolog", "lantus", "levemir", "humulin", "tresiba", "apidra", "basaglar", "toujeo"])
    metformin = _patients_on(["metformin"])
    sulf = _patients_on(["glipizide", "glyburide", "glimepiride", "glibenclamide"])
    glp1 = _patients_on(["liraglutide", "dulaglutide", "semaglutide", "exenatide", "lixisenatide",
                          "victoza", "trulicity", "ozempic", "rybelsus", "byetta", "bydureon"])
    sglt2 = _patients_on(["empagliflozin", "canagliflozin", "dapagliflozin", "ertugliflozin",
                           "jardiance", "invokana", "farxiga", "steglatro"])
    dpp4 = _patients_on(["sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
                          "januvia", "onglyza", "tradjenta"])

    out = cohort.copy()
    out["on_insulin"] = out["subject_id"].isin(insulin)
    out["on_metformin"] = out["subject_id"].isin(metformin)
    out["on_sulfonylurea"] = out["subject_id"].isin(sulf)
    out["on_glp1"] = out["subject_id"].isin(glp1)
    out["on_sglt2"] = out["subject_id"].isin(sglt2)
    out["on_dpp4"] = out["subject_id"].isin(dpp4)
    out["n_antidm_classes"] = out[
        ["on_insulin", "on_metformin", "on_sulfonylurea",
         "on_glp1", "on_sglt2", "on_dpp4"]
    ].sum(axis=1).astype(int)
    return out


# ---------------------------------------------------------------
# 7. Demo
# ---------------------------------------------------------------
def main():
    tables, src = load_mimic()
    print(f"Data source: {src}")
    for k, v in tables.items():
        if not k.startswith("_"):
            print(f"  {k:15s}: {len(v):,} rows")

    cohort = build_cohort(tables)
    cohort = enrich_with_a1c(cohort, tables)
    cohort = engineer_features(cohort, tables["prescriptions"])

    print(f"\nCohort size: {cohort['is_diabetic'].sum():,} / {len(cohort):,} patients")
    print("\nFlag overlap:")
    print(cohort.groupby(
        ["has_multi_admit_dm", "has_primary_dm", "has_antidm_rx"]
    )["subject_id"].count().rename("n"))

    if "_truth" in tables:
        truth = tables["_truth"]
        ev = cohort.merge(truth, on="subject_id")
        tp = ((ev.is_diabetic) & (ev.is_diabetic_truth)).sum()
        fp = ((ev.is_diabetic) & (~ev.is_diabetic_truth)).sum()
        fn = ((~ev.is_diabetic) & (ev.is_diabetic_truth)).sum()
        tn = ((~ev.is_diabetic) & (~ev.is_diabetic_truth)).sum()
        s = tp / (tp + fn) if tp + fn else 0
        sp = tn / (tn + fp) if tn + fp else 0
        ppv = tp / (tp + fp) if tp + fp else 0
        print(f"\nEvaluation (synthetic truth):")
        print(f"  sensitivity={s:.3f}  specificity={sp:.3f}  PPV={ppv:.3f}")

    return cohort


if __name__ == "__main__":
    main()
