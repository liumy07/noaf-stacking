# Analysis code: Interpretable stacking ensemble model for new-onset atrial fibrillation in critically ill patients

This repository contains the complete analysis code for the manuscript
*"Development and external validation of an interpretable stacking ensemble model for new-onset atrial fibrillation in critically ill patients"*
(submitted to **Journal of Translational Medicine**, 2026).

The code reproduces every table and figure in the manuscript from the two public
source databases — [MIMIC-IV (v3.1)](https://physionet.org/content/mimiciv/3.1/)
and the [eICU Collaborative Research Database (v2.0)](https://physionet.org/content/eicu-crd/2.0/).
**No patient-level data are included in this repository**; source extracts are governed by the PhysioNet Data Use Agreement and are not redistributed.

## Repository structure

| Folder | Contents |
|---|---|
| `R_cohort_construction/` | R scripts that build the pre-imputation analysis cohorts from raw MIMIC-IV / eICU-CRD extracts (34,384 and 32,067 ICU stays) |
| `00_common/` | Shared Python utilities (variable harmonisation, feature engineering) |
| `01_core_pipeline_v90/` | Final leakage-free pipeline: fold-wise predictor screening (univariable logistic regression + LASSO) re-run inside every cross-validation fold, MissForest imputation fitted on training data only, stacking of the three best base learners (MLP, XGBoost, LightGBM) with a LASSO meta-learner |
| `02_results_and_figures/` | Performance evaluation (AUC/PR, calibration, decision-curve analysis, DeLong tests, bootstrap CIs), SHAP interpretation, and generation of all manuscript figures |
| `03_manuscript/` | Scripts that assemble the manuscript tables, supplementary materials and the TRIPOD+AI checklist |
| `04_deck/` | Scripts for the presentation deck (not part of the manuscript) |

## Software requirements

- **R** ≥ 4.1 (cohort construction)
- **Python** 3.13 with `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `missforest`, `matplotlib`, `pandas`, `numpy`

## How to run

1. Obtain credentialed access to MIMIC-IV v3.1 and eICU-CRD v2.0 via [PhysioNet](https://physionet.org).
2. Edit `DATA_DIR` / `OUT_DIR` at the top of the R scripts in `R_cohort_construction/` and run them to produce the pre-imputation cohort files.
3. Run `01_core_pipeline_v90/step53d_v90_final.py` to train the stacking ensemble and generate out-of-fold predictions.
4. Run the scripts in `02_results_and_figures/` to reproduce all performance metrics, calibration plots, decision-curve analyses and SHAP figures.

## Data availability

MIMIC-IV (v3.1) and eICU-CRD (v2.0) are publicly available to credentialed users through PhysioNet (https://physionet.org). Derived analysis files cannot be redistributed under the data-use agreements.

## License

Released under the [MIT License](LICENSE).
