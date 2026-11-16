# -*- coding: utf-8 -*-
"""
noaf_common_v70.py — leak-free imputation branch (v70)

WHY THIS FILE EXISTS
--------------------
The original cleaning script (01_clean.R) ran

    clear_data <- missForest(clear_data)$ximp      # line 178 (MIMIC)
    clear_data <- missForest(clear_data)$ximp      # line 335 (eICU)

on the WHOLE cohort, before any train/test split.  Every downstream model was
therefore trained on values whose conditional means had been estimated with
test-set rows included — a textbook pre-processing leak.

v70 rebuilds the same analysis with the order reversed:

    1. split MIMIC-IV 70/30 (stratified, seed 42)   <-- FIRST
    2. fit the imputer on the training rows ONLY    <-- THEN
    3. transform train / internal test / eICU with that fitted imputer

Nothing else changes: same 39 variables, same univariable screen, same LASSO
1-SE rule, same nine learners with identical hyper-parameters, same 10-fold CV,
same seed, same LASSO meta-learner.

VARIABLE TYPES
--------------
`01_clean.R` converts the binary covariates to factors before missForest, so the
imputed CSV stores hypertension/diabetes/... as integer 0/1 while Gender,
Vasopressors and Antibiotic stay as "Male"/"Yes".  We restore one consistent
dictionary: every binary predictor is mapped to 0/1 *before* imputation (so the
imputer never invents an intermediate value), and the categorical block keeps
its string levels for the one-hot encoder.

IMPUTER
-------
missForest = random-forest chained equations.  sklearn has no identical
implementation, so we use IterativeImputer with a RandomForestRegressor and a
single imputation pass (max_iter=1, i.e. one imputation of each variable on the
observed data), then round binary columns back to 0/1.  This is the standard
chained-equations equivalent; the manuscript wording should say
"random-forest-based chained equations (missForest-type)".  Setting
MISSFOREST_ITER > 1 gives the full iterative missForest behaviour.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# paths  (v70 = TRUE raw pre-imputation data, real missingness restored)
# ----------------------------------------------------------------------
DATA_DIR = Path(r"D:/BJND-4002-11项目/BJND-4002-11")
MIMIC_CSV = DATA_DIR / "插补前mimic_data.csv"
EICU_CSV = DATA_DIR / "插补前eicu_data.csv"

REV_DIR = Path(r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/rev_noaf")
V70_DIR = REV_DIR / "v70_leakfree"
FIG_DIR = V70_DIR / "figures"
TAB_DIR = V70_DIR / "tables"
for _d in (V70_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 10
MISSFOREST_ITER = 10          # chained-equation passes; missForest default is 10
RF_TREES = 100                 # ntrees in missForest's default forest

# ----------------------------------------------------------------------
# harmonised 39 variables (identical to v18 / v40)
# ----------------------------------------------------------------------
VARS39 = [
    "Age", "Gender", "weight",
    "hypertension", "diabetes", "congestive_heart_failure",
    "mild_liver_disease", "severe_liver_disease", "renal_disease",
    "cerebrovascular_disease", "sepsis", "InvasiveVent",
    "output_urine_first",
    "heart_rate_max", "heart_rate_min", "resp_rate_max", "resp_rate_min",
    "wbc_max", "wbc_min", "platelet_max", "platelet_min",
    "creatinine_max", "creatinine_min", "bun_max", "bun_min",
    "aniongap_max", "aniongap_min", "calcium_total_max", "calcium_total_min",
    "glucose_max", "glucose_min", "potassium_max", "potassium_min",
    "sodium_max", "sodium_min", "hemoglobin_max", "hemoglobin_min",
    "Vasopressors", "Antibiotic",
]
assert len(VARS39) == 39

# binary predictors stored as integer 0/1 after the R factor conversion
BIN01 = ["hypertension", "diabetes", "congestive_heart_failure",
         "mild_liver_disease", "severe_liver_disease", "renal_disease",
         "cerebrovascular_disease", "sepsis", "InvasiveVent"]

# binary predictors stored as strings in the CSV
BIN_STR = {"Gender": {"Male": 1, "Female": 0},
           "Vasopressors": {"Yes": 1, "No": 0},
           "Antibiotic": {"Yes": 1, "No": 0}}

CONT = [v for v in VARS39 if v not in BIN01 and v not in BIN_STR]

LABELS = {
    "Age": "Age (years)", "Gender": "Male gender", "weight": "Weight (kg)",
    "hypertension": "Hypertension", "diabetes": "Diabetes",
    "congestive_heart_failure": "Congestive heart failure",
    "mild_liver_disease": "Mild liver disease",
    "severe_liver_disease": "Severe liver disease",
    "renal_disease": "Renal disease",
    "cerebrovascular_disease": "Cerebrovascular disease",
    "sepsis": "Sepsis", "InvasiveVent": "Invasive ventilation",
    "output_urine_first": "Urine output, first 24 h (per 100 mL)",
    "heart_rate_max": "Heart rate, maximum (bpm)",
    "heart_rate_min": "Heart rate, minimum (bpm)",
    "resp_rate_max": "Respiratory rate, maximum",
    "resp_rate_min": "Respiratory rate, minimum",
    "wbc_max": "WBC, maximum (10\u2079/L)", "wbc_min": "WBC, minimum (10\u2079/L)",
    "platelet_max": "Platelet count, maximum (10\u2079/L)",
    "platelet_min": "Platelet count, minimum (10\u2079/L)",
    "creatinine_max": "Creatinine, maximum (mg/dL)",
    "creatinine_min": "Creatinine, minimum (mg/dL)",
    "bun_max": "BUN, maximum (mg/dL)", "bun_min": "BUN, minimum (mg/dL)",
    "aniongap_max": "Anion gap, maximum (mEq/L)",
    "aniongap_min": "Anion gap, minimum (mEq/L)",
    "calcium_total_max": "Calcium, maximum (mg/dL)",
    "calcium_total_min": "Calcium, minimum (mg/dL)",
    "glucose_max": "Glucose, maximum (mg/dL)",
    "glucose_min": "Glucose, minimum (mg/dL)",
    "potassium_max": "Potassium, maximum (mEq/L)",
    "potassium_min": "Potassium, minimum (mEq/L)",
    "sodium_max": "Sodium, maximum (mEq/L)",
    "sodium_min": "Sodium, minimum (mEq/L)",
    "hemoglobin_max": "Hemoglobin, maximum (g/dL)",
    "hemoglobin_min": "Hemoglobin, minimum (g/dL)",
    "Vasopressors": "Vasopressor use", "Antibiotic": "Antibiotic use",
}

EICU_RENAME = {"sepsis3": "sepsis", "invasiveVent": "InvasiveVent",
               "output_urine_avg": "output_urine_first"}

MODEL_NAMES = ["MLP", "LightGBM", "XGBoost", "RF", "Logistic",
               "DT", "SVM", "Ridge", "KNN"]
DISPLAY = {"MLP": "Multilayer perceptron", "LightGBM": "LightGBM",
           "XGBoost": "XGBoost", "RF": "Random forest",
           "Logistic": "Logistic regression", "DT": "Decision tree",
           "SVM": "Support vector machine", "Ridge": "Ridge regression",
           "KNN": "K-nearest neighbours", "Stacking": "Stacking ensemble"}

# consistent colours across every figure of the revision
COLORS = {"Stacking": "#B2182B", "MLP": "#2166AC", "XGBoost": "#1B7837",
          "RF": "#8073AC", "LightGBM": "#E08214", "Logistic": "#666666",
          "DT": "#999999", "SVM": "#8C510A", "Ridge": "#01665E",
          "KNN": "#BDBDBD"}
LS = {"Stacking": "-", "MLP": "-", "XGBoost": "-", "RF": "-", "LightGBM": "--"}


# ----------------------------------------------------------------------
# loading / harmonising
# ----------------------------------------------------------------------
def _harmonise(df: pd.DataFrame) -> pd.DataFrame:
    """Map the two binary string columns to 0/1 so all predictors are numeric."""
    out = pd.DataFrame(index=df.index)
    for v in VARS39:
        s = df[v]
        if v in BIN_STR:
            mp = BIN_STR[v]
            out[v] = s.astype(str).str.strip().map(mp)
            if out[v].isna().any():
                # tolerate already-numeric encodings
                num = pd.to_numeric(s, errors="coerce")
                if num.notna().sum() == s.notna().sum():
                    out[v] = num
        else:
            out[v] = pd.to_numeric(s, errors="coerce")
    return out


def load_raw_cohorts():
    """Raw (un-imputed) harmonised predictor frames + outcomes.

    NOTE: `mimic_data.csv` was written AFTER missForest ran in 01_clean.R, so it
    has no NaN.  We therefore cannot recover the per-cell missingness pattern
    from the file alone.  The leakage that matters for model development is the
    one introduced by fitting the imputer on all rows; v70 removes it by
    re-fitting the imputer on the training rows only.  See
    `_inject_realistic_mask()` for the sensitivity analysis that recreates a
    missForest-era mask to quantify how much the leak was actually worth.
    """
    m = pd.read_csv(MIMIC_CSV, encoding="utf-8-sig")
    e = pd.read_csv(EICU_CSV, encoding="utf-8-sig").rename(columns=EICU_RENAME)

    y_m = m["NOAF"].map({"Yes": 1, "No": 0}).astype(int)
    y_e = e["NOAF"].map({"Yes": 1, "No": 0}).astype(int)

    X_m = _harmonise(m)
    X_e = _harmonise(e)
    return X_m, y_m, X_e, y_e


def split_indices(y):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=0.3, stratify=y, random_state=SEED)


# ----------------------------------------------------------------------
# leak-free imputation
# ----------------------------------------------------------------------
def fit_imputer(X_train: pd.DataFrame):
    """Fit a missForest-type imputer on TRAINING ROWS ONLY."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import IterativeImputer

    est = RandomForestRegressor(n_estimators=RF_TREES, max_features="sqrt",
                                min_samples_leaf=3, n_jobs=-1,
                                random_state=SEED)
    imp = IterativeImputer(estimator=est, max_iter=MISSFOREST_ITER,
                           tol=1e-3, initial_strategy="median",
                           imputation_order="ascending",
                           skip_complete=True, random_state=SEED,
                           sample_posterior=False)
    imp.fit(X_train)
    return imp


def apply_imputer(imp, X: pd.DataFrame) -> pd.DataFrame:
    Z = pd.DataFrame(imp.transform(X), index=X.index, columns=X.columns)
    # binary variables must stay binary after rounding
    for c in BIN01 + list(BIN_STR):
        if c in Z:
            Z[c] = Z[c].round().clip(0, 1).astype(int)
    return Z


def _inject_realistic_mask(df: pd.DataFrame, rate_by_col: dict, seed=SEED):
    """Recreate a plausible pre-imputation missingness pattern (sensitivity only)."""
    rng = np.random.default_rng(seed)
    Z = df.copy()
    for c, r in rate_by_col.items():
        if c not in Z or r <= 0:
            continue
        mask = rng.random(len(Z)) < r
        Z.loc[mask, c] = np.nan
    return Z


# ----------------------------------------------------------------------
# preprocessing / models  (byte-identical to v40)
# ----------------------------------------------------------------------
def make_preprocessor(num_cols, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False,
                                               min_frequency=5))]), cat_cols),
    ], remainder="drop")


def split_columns(X):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def base_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    import xgboost as xgb
    import lightgbm as lgb
    return {
        "RF":       RandomForestClassifier(n_estimators=1000, max_features="sqrt",
                                           min_samples_leaf=5, n_jobs=-1,
                                           random_state=SEED),
        "XGBoost":  xgb.XGBClassifier(n_estimators=1000, max_depth=6,
                                      learning_rate=0.03, subsample=0.8,
                                      colsample_bytree=0.8, min_child_weight=4,
                                      reg_alpha=0.1, reg_lambda=1.0,
                                      objective="binary:logistic",
                                      eval_metric="auc", tree_method="hist",
                                      n_jobs=-1, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(n_estimators=1200, num_leaves=63,
                                       learning_rate=0.03, subsample=0.8,
                                       colsample_bytree=0.8, min_child_samples=20,
                                       reg_alpha=0.1, reg_lambda=1.0,
                                       objective="binary", metric="auc",
                                       n_jobs=-1, random_state=SEED, verbose=-1),
        "MLP":      MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                  solver="adam", alpha=1e-3, batch_size=512,
                                  max_iter=80, early_stopping=True,
                                  validation_fraction=0.1, random_state=SEED),
        "SVM":      SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000,
                        random_state=SEED),
        "DT":       DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                           random_state=SEED),
        "Logistic": LogisticRegression(penalty="l2", C=0.1, max_iter=2000,
                                       random_state=SEED),
        "Ridge":    RidgeClassifierCV(alphas=np.logspace(-3, 3, 25)),
        "KNN":      KNeighborsClassifier(n_neighbors=50, n_jobs=-1),
    }


def get_scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------
def fast_auc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    sv = s[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def average_precision(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def net_benefit(y, p, thresholds):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(y)
    out = np.empty(len(thresholds), dtype=float)
    for i, pt in enumerate(thresholds):
        pred = p >= pt
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        out[i] = tp / n - fp / n * (pt / (1 - pt))
    return out


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def calib_slope_intercept(y, p):
    import statsmodels.api as sm
    y = np.asarray(y, dtype=float)
    lp = _logit(p)
    res = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    return float(res.params[0]), float(res.params[1]), float(res.bse[0]), float(res.bse[1])


# ----------------------------------------------------------------------
# plotting defaults — SCI ready
# ----------------------------------------------------------------------
def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "axes.grid": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return plt


def save_fig(fig, stem, dirs=(FIG_DIR,)):
    outs = []
    for d in dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            fp = d / f"{stem}.{ext}"
            fig.savefig(fp)
            outs.append(fp)
    return outs


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
