# -*- coding: utf-8 -*-
"""
step41_v80_ci.py — every confidence interval needed for the revised manuscript,
computed on the v80 frozen-predictor scores.

Outputs (v80_foldwise/)
    v80_ci_auc.csv          AUC + DeLong 95% CI  (10 models x 3 cohorts)
    v80_ci_ap.csv           AP + bootstrap 95% CI (10 models x 3 cohorts)
    v80_ci_nb.csv           net benefit @0.10/0.20 + bootstrap CI (top4 x 2)
    v80_ci_calibration.csv  intercept/slope + bootstrap CI (stacking, 2 cohorts)
    v80_delong_paired.csv   paired DeLong, stacking vs each base learner
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR, fast_auc, average_precision  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from scipy.stats import norm  # noqa: E402

SEED, NBOOT = 42, 1000
NINE = ["MLP", "XGBoost", "RF", "LightGBM", "Logistic", "SVM", "Ridge",
        "DT", "KNN"]
D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
TOP3 = [str(x) for x in D["top3"]]
Y = {"train": D["y_train"].astype(int), "test": D["y_test"].astype(int),
     "eICU": D["y_eicu"].astype(int)}
PFX = {"train": "oof__", "test": "tst__", "eICU": "ext__"}
ALL = NINE + ["Stacking"]


def boot_idx(y, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos), replace=True),
                           rng.choice(neg, len(neg), replace=True)])


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ------------------------------------------------------------- AUC DeLong CI
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_ci(y, s):
    y = np.asarray(y, dtype=int)
    m = int(y.sum())
    n = len(y) - m
    order = np.argsort(-y)
    p = np.asarray(s, dtype=float)[order]
    tz, tx, ty = _midrank(p), _midrank(p[:m]), _midrank(p[m:])
    auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    se = float(np.sqrt(np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n))
    z = norm.ppf(0.975)
    return float(auc), float(auc - z * se), float(auc + z * se)


rows = []
for m in ALL:
    for coh in ("train", "test", "eICU"):
        auc, lo, hi = delong_ci(Y[coh], D[PFX[coh] + m])
        rows.append(dict(Model=m, cohort=coh, n=int(len(Y[coh])),
                         AUC=auc, CI_lo=lo, CI_hi=hi))
        print(f"AUC {m:9s} {coh:5s} {auc:.4f} ({lo:.4f}-{hi:.4f})", flush=True)
pd.DataFrame(rows).to_csv(V80_DIR / "v80_ci_auc.csv", index=False,
                          encoding="utf-8-sig")

# ------------------------------------------------------------------ AP CI
rows = []
for m in ALL:
    for coh in ("train", "test", "eICU"):
        y, s = Y[coh], D[PFX[coh] + m]
        rng = np.random.default_rng(SEED)
        aps = np.empty(NBOOT)
        for b in range(NBOOT):
            ix = boot_idx(y, rng)
            aps[b] = average_precision(y[ix], s[ix])
        lo, hi = np.percentile(aps, [2.5, 97.5])
        rows.append(dict(Model=m, cohort=coh, AP=average_precision(y, s),
                         lo=lo, hi=hi))
        print(f"AP  {m:9s} {coh:5s} {rows[-1]['AP']:.4f} "
              f"({lo:.4f}-{hi:.4f})", flush=True)
pd.DataFrame(rows).to_csv(V80_DIR / "v80_ci_ap.csv", index=False,
                          encoding="utf-8-sig")

# ------------------------------------------------------------------ NB CI
MODELS4 = ["Stacking"] + TOP3
rows = []
for m in MODELS4:
    for coh in ("test", "eICU"):
        y, s = Y[coh], D[PFX[coh] + m]
        rng = np.random.default_rng(SEED + 1)
        for thr in (0.10, 0.20):
            def nb_of(yy, ss):
                pred = (ss >= thr).astype(int)
                tp = np.sum((pred == 1) & (yy == 1))
                fp = np.sum((pred == 1) & (yy == 0))
                n = len(yy)
                return tp / n - fp / n * thr / (1 - thr)
            boots = np.empty(NBOOT)
            for b in range(NBOOT):
                ix = boot_idx(y, rng)
                boots[b] = nb_of(y[ix], s[ix])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(Model=m, cohort=coh, thr=thr, nb=nb_of(y, s),
                             lo=lo, hi=hi))
            print(f"NB  {m:9s} {coh:5s} @{thr:.2f} {rows[-1]['nb']:+.4f} "
                  f"({lo:+.4f}-{hi:+.4f})", flush=True)
pd.DataFrame(rows).to_csv(V80_DIR / "v80_ci_nb.csv", index=False,
                          encoding="utf-8-sig")

# ------------------------------------------------- calibration CI (joint fit)
rows = []
for coh in ("test", "eICU"):
    y, s = Y[coh], D[PFX[coh] + "Stacking"]
    lp = logit(s).astype(float)

    def fit_cal(yy, lpp):
        # logistic recalibration: intercept and slope estimated JOINTLY
        lr = LogisticRegression(C=1e6, max_iter=500).fit(
            lpp.reshape(-1, 1), yy)
        return float(lr.intercept_[0]), float(lr.coef_[0, 0])

    i0, s0 = fit_cal(y, lp)
    rng = np.random.default_rng(SEED + 2)
    out = np.empty((NBOOT, 2))
    for b in range(NBOOT):
        ix = boot_idx(y, rng)
        out[b] = fit_cal(y[ix], lp[ix])
    i_lo, i_hi = np.percentile(out[:, 0], [2.5, 97.5])
    s_lo, s_hi = np.percentile(out[:, 1], [2.5, 97.5])
    brier = float(np.mean((s - y) ** 2))
    eo = float(np.mean(s) / np.mean(y))
    rows.append(dict(cohort=coh, intercept=i0, i_lo=i_lo, i_hi=i_hi,
                     slope=s0, s_lo=s_lo, s_hi=s_hi, brier=brier, E_O=eo))
    print(f"CAL {coh:5s} int {i0:+.3f} ({i_lo:+.3f}-{i_hi:+.3f})  "
          f"slope {s0:.3f} ({s_lo:.3f}-{s_hi:.3f})  E:O {eo:.3f}", flush=True)
pd.DataFrame(rows).to_csv(V80_DIR / "v80_ci_calibration.csv", index=False,
                          encoding="utf-8-sig")

# ------------------------------------------------------------- paired DeLong
def delong_paired(y, sa, sb):
    y = np.asarray(y, int)
    m = int(y.sum())
    n = len(y) - m
    order = np.argsort(-y)
    pa = np.asarray(sa, float)[order]
    pb = np.asarray(sb, float)[order]
    tza, tzb = _midrank(pa), _midrank(pb)
    txa, txb = _midrank(pa[:m]), _midrank(pb[:m])
    tya, tyb = _midrank(pa[m:]), _midrank(pb[m:])
    auc_a = tza[:m].sum() / m / n - (m + 1.0) / 2 / n
    auc_b = tzb[:m].sum() / m / n - (m + 1.0) / 2 / n
    v01a, v01b = (tza[:m] - txa) / n, (tzb[:m] - txb) / n
    v10a, v10b = 1 - (tza[m:] - tya) / m, 1 - (tzb[m:] - tyb) / m
    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    cov = s01 / m + s10 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = float(np.sqrt(max(var, 1e-18)))
    delta = float(auc_a - auc_b)
    z = delta / se
    p = 2 * (1 - norm.cdf(abs(z)))
    zc = norm.ppf(0.975)
    return delta, delta - zc * se, delta + zc * se, p


rows = []
for coh in ("test", "eICU"):
    y = Y[coh]
    for m in NINE:
        d, lo, hi, p = delong_paired(y, D[PFX[coh] + "Stacking"],
                                     D[PFX[coh] + m])
        rows.append(dict(cohort=coh, base=m, delta=d, lo=lo, hi=hi, p=p))
        print(f"DL  {coh:5s} vs {m:9s} {d:+.4f} ({lo:+.4f}-{hi:+.4f}) "
              f"p={p:.2e}", flush=True)
pd.DataFrame(rows).to_csv(V80_DIR / "v80_delong_paired.csv", index=False,
                          encoding="utf-8-sig")
print("[DONE]")
