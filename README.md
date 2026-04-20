# Medical Code Systems in Practice — A Type 2 Diabetes Case Study

Reproducible code that accompanies the Medium post *"Medical Code Systems in Practice: A Type 2 Diabetes Case Study."* The post treats Type 2 diabetes phenotyping as a single applied use case for the major medical code systems — **ICD-10-CM**, **ICD-9-CM**, **RxNorm**, **LOINC**, and **CPT** — and walks the same task across three settings: synthetic administrative claims, real MIMIC-IV hospital EHR, and the PheKB / eMERGE reference library.

## What's in this repo

```
medical-code-phenotyping/
├── notebooks/
│   ├── Diabetic_Cohort_Notebook.ipynb     # Case Study A — synthetic CMS-style claims
│   └── Diabetic_Cohort_MIMIC.ipynb        # Case Study B — MIMIC-IV hospital EHR
├── src/
│   ├── diabetic_cohort.py                 # Phenotype primitives for claims data
│   └── mimic_diabetes.py                  # Phenotype primitives for MIMIC-IV
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/medical-code-phenotyping.git
cd medical-code-phenotyping

# 2. Install dependencies (Python 3.10+ recommended)
python -m pip install -r requirements.txt

# 3. Launch the notebooks
jupyter lab
```

Then open either notebook under `notebooks/`.

### Case Study A — Synthetic claims

`Diabetic_Cohort_Notebook.ipynb` runs end-to-end with no external data. It ships a synthetic generator that mirrors the CMS Limited Data Set (LDS) table grain (patients, diagnosis_claims, pharmacy_claims, lab_results, procedure_claims) and exercises ICD-10-CM, RxNorm, LOINC, and CPT together.

### Case Study B — MIMIC-IV

`Diabetic_Cohort_MIMIC.ipynb` auto-detects whether real MIMIC-IV CSVs are present and falls back to a schema-faithful synthetic fixture when they are not.

To run against **real MIMIC-IV**, download one of:

- **Demo (no credentialing, ~125 MB):** [MIMIC-IV Clinical Database Demo v2.2](https://physionet.org/content/mimic-iv-demo/2.2/)
- **Full (credentialed):** [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) — requires PhysioNet account, CITI "Data or Specimens Only Research" training, and a signed data-use agreement.

Set the `MIMIC_ROOT` environment variable (or the `MIMIC_ROOT` constant at the top of the notebook) to the folder that contains `hosp/`:

```bash
# macOS / Linux
export MIMIC_ROOT=./physionet.org/files/mimic-iv-demo/2.2

# Windows PowerShell
$env:MIMIC_ROOT = ".\physionet.org\files\mimic-iv-demo\2.2"
```

## Code-system primitives

The core of each module is a small set of composable functions that can be dropped into any phenotyping pipeline with minimal glue:

- `filter_diabetes_diagnoses` / `is_diabetes_icd` — ICD prefix filter (handles ICD-9 ↔ ICD-10 coexistence in MIMIC-IV).
- `patients_with_outpatient_dx` / `patients_with_multi_admission_dm` / `patients_with_primary_dm` — Dx-based patient-set criteria.
- `patients_with_antidm_rx` — RxNorm set membership (claims) or lowercase token matching against free-text drug names (MIMIC-IV).
- `enrich_with_a1c` — LOINC-aware A1c aggregation with label-keyword fallback.
- `engineer_features` — medication-class intensity features (metformin, insulin, GLP-1, SGLT-2, DPP-4, sulfonylurea).
- `build_cohort` — OR-combined phenotype with an optional strict (any 2 of 3) variant for ICU-heavy data.

Swap the diabetes code lists for another condition and the structure works unchanged.

## Key code systems exercised

| System       | Role in the pipeline                                               |
|--------------|---------------------------------------------------------------------|
| ICD-10-CM    | Post-2015 diagnoses: E10–E13 family for diabetes                    |
| ICD-9-CM     | Pre-2015 diagnoses in MIMIC-IV: 250.\* family                        |
| RxNorm       | Antidiabetic medications in claims data                             |
| LOINC        | HbA1c laboratory observations (4548-4, 17856-6, 41995-2)            |
| CPT          | Procedure audit that A1c was actually billed (83036)                |

## References

- Kho AN, Hayes MG, Rasmussen-Torvik L, et al. "Use of diverse electronic medical record systems to identify genetic risk for type 2 diabetes within a genome-wide association study." *J Am Med Inform Assoc.* 2012;19(2):212–218.
- PheKB — Phenotype KnowledgeBase. https://phekb.org/
- MIMIC-IV — Johnson A, et al. MIT Laboratory for Computational Physiology. https://physionet.org/content/mimiciv/

## License

MIT — see `LICENSE`.

## Disclaimer

This repository is for educational and research purposes. Nothing here is a validated clinical decision-support tool. Any use on real patient data must go through the appropriate data-use agreements, IRB review, and clinical validation.
