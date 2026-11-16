# -*- coding: utf-8 -*-
"""
step27_v80_nine_models.py — Nine base learners, ROC + PR on the training set
(10-fold out-of-fold) and the internal hold-out test set, v70 leak-free.

Outputs (v80_leakfree/)
    v80_nine_models_metrics.csv    AUC (DeLong 95% CI) + AP, OOF / test / eICU
    v40_vs_v80_all_models.csv      manuscript (v40) vs v70 comparison
    figures/Fig3_NineModels_TrainTest.{png,pdf,svg}   2x2 A-D
    figures/FigS1_TrainOOF_ROC_PR.{png,pdf,svg}       2-panel group
    figures/FigS1S_TrainOOF_ROC / _PR singles for slides
    figures/Fig4CG_PR_two_cohorts.{png,pdf,svg}       wide 2-panel (slide 15)

OVERLAP POLICY — identical to step22: AUC/AP values live in the legend, the
legend sits completely outside the axes (below), and reference lines are the
only extra artists.  Nothing is ever drawn across a curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import (  # noqa: E402
    V80_DIR, FIG_DIR, COLORS, set_style, save_fig, fast_auc, average_precision,
)

plt = set_style()
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_curve, precision_recall_curve  # noqa: E402

NINE = ["MLP", "XGBoost", "RF", "LightGBM", "Logistic", "SVM", "Ridge",
        "DT", "KNN"]
FULL = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
        "RF": "Random forest", "LightGBM": "LightGBM",
        "Logistic": "Logistic regression", "SVM": "Support vector machine",
        "Ridge": "Ridge regression", "DT": "Decision tree",
        "KNN": "K-nearest neighbours"}
LW = 1.25

D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
Y = {"train": D["y_train"].astype(int), "test": D["y_test"].astype(int),
     "eICU": D["y_eicu"].astype(int)}
PFX = {"train": "oof__", "test": "tst__", "eICU": "ext__"}
NTITLE = {"train": "Training set — 10-fold out-of-fold (n = 24,068)",
          "test": "Internal hold-out test set (n = 10,316)",
          "eICU": "eICU-CRD external cohort (n = 32,067)"}

# ----------------------------------------------------------------------
# DeLong 95% CI (same estimator family as the manuscript's step3_M08)
# ----------------------------------------------------------------------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_ci(y, s):
    """AUC and its DeLong SE -> (auc, lo, hi)."""
    y = np.asarray(y, dtype=int)
    m = int(y.sum())
    n = len(y) - m
    order = np.argsort(-y)                       # positives first
    preds = np.asarray(s, dtype=float)[order][None, :]
    tz = _midrank(preds[0])
    tx = _midrank(preds[0, :m])
    ty = _midrank(preds[0, m:])
    auc = tz[:m].sum() / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    se = float(np.sqrt(np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n))
    z = norm.ppf(0.975)
    return float(auc), float(auc - z * se), float(auc + z * se)


# ----------------------------------------------------------------------
# metrics + comparison with the manuscript (v40, full-cohort imputation)
# ----------------------------------------------------------------------
V40 = {  # manuscript Table 2 (AUC strings)
    ("Logistic", "train"): "0.833 (0.828-0.839)",
    ("Ridge", "train"): "0.829 (0.823-0.834)",
    ("DT", "train"): "0.808 (0.803-0.814)",
    ("KNN", "train"): "0.803 (0.798-0.809)",
    ("SVM", "train"): "0.822 (0.816-0.828)",
    ("MLP", "train"): "0.838 (0.833-0.843)",
    ("RF", "train"): "0.835 (0.830-0.841)",
    ("LightGBM", "train"): "0.831 (0.826-0.837)",
    ("XGBoost", "train"): "0.835 (0.830-0.841)",
    ("Stacking", "train"): "0.842 (0.837-0.847)",
    ("Logistic", "test"): "0.833 (0.824-0.841)",
    ("Ridge", "test"): "0.827 (0.819-0.835)",
    ("DT", "test"): "0.829 (0.821-0.837)",
    ("KNN", "test"): "0.803 (0.794-0.812)",
    ("SVM", "test"): "0.826 (0.817-0.834)",
    ("MLP", "test"): "0.844 (0.835-0.852)",
    ("RF", "test"): "0.838 (0.831-0.847)",
    ("LightGBM", "test"): "0.841 (0.832-0.849)",
    ("XGBoost", "test"): "0.840 (0.832-0.849)",
    ("Stacking", "test"): "0.846 (0.838-0.854)",
    ("Logistic", "eICU"): "0.741 (0.731-0.750)",
    ("Ridge", "eICU"): "0.735 (0.726-0.744)",
    ("DT", "eICU"): "0.741 (0.731-0.750)",
    ("KNN", "eICU"): "0.706 (0.697-0.716)",
    ("SVM", "eICU"): "0.736 (0.726-0.746)",
    ("MLP", "eICU"): "0.755 (0.746-0.765)",
    ("RF", "eICU"): "0.745 (0.735-0.755)",
    ("LightGBM", "eICU"): "0.755 (0.745-0.764)",
    ("XGBoost", "eICU"): "0.755 (0.745-0.763)",
    ("Stacking", "eICU"): "0.757 (0.747-0.767)",
}

rows = []
for m in NINE + ["Stacking"]:
    for coh in ("train", "test", "eICU"):
        s = D[PFX[coh] + m]
        auc, lo, hi = delong_ci(Y[coh], s)
        rows.append(dict(Model=m, cohort=coh, n=int(len(Y[coh])),
                         AUC=round(auc, 4), CI_lo=round(lo, 4),
                         CI_hi=round(hi, 4),
                         AP=round(average_precision(Y[coh], s), 4),
                         AUC_CI=f"{auc:.3f} ({lo:.3f}-{hi:.3f})"))
MET = pd.DataFrame(rows)
MET.to_csv(V80_DIR / "v80_nine_models_metrics.csv", index=False,
           encoding="utf-8-sig")
print(MET.to_string(index=False))

cmp_rows = []
for m in NINE + ["Stacking"]:
    rec = {"Model": m}
    for coh in ("train", "test", "eICU"):
        v70 = float(MET[(MET.Model == m) & (MET.cohort == coh)].AUC.iloc[0])
        v40 = float(V40[(m, coh)].split(" ")[0])
        rec[f"{coh}_v40"] = V40[(m, coh)]
        rec[f"{coh}_v70"] = f"{v70:.3f} ({MET[(MET.Model == m) & (MET.cohort == coh)].CI_lo.iloc[0]:.3f}-{MET[(MET.Model == m) & (MET.cohort == coh)].CI_hi.iloc[0]:.3f})"
        rec[f"{coh}_delta"] = round(v70 - v40, 4)
    cmp_rows.append(rec)
CMP = pd.DataFrame(cmp_rows)
CMP.to_csv(V80_DIR / "v40_vs_v80_all_models.csv", index=False,
           encoding="utf-8-sig")
print("\n", CMP.to_string(index=False))


# ----------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------
def draw_roc(ax, cohort, models, letter):
    y = Y[cohort]
    order = sorted(models, key=lambda m: fast_auc(y, D[PFX[cohort] + m]),
                   reverse=True)
    for m in order:
        fpr, tpr, _ = roc_curve(y, D[PFX[cohort] + m])
        ax.plot(fpr, tpr, color=COLORS[m], lw=LW, ls="-",
                solid_capstyle="round", zorder=2)
    ax.plot([0, 1], [0, 1], color="#B0B0B0", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_xlabel("1 \u2212 specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{letter}   {NTITLE[cohort]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    handles = [Line2D([], [], color="#B0B0B0", lw=0.8, ls=(0, (4, 3)),
                      label="Chance (AUC 0.500)")]
    handles += [Line2D([], [], color=COLORS[m], lw=LW,
                       label=f"{FULL[m]}  ({fast_auc(y, D[PFX[cohort] + m]):.3f})")
                for m in order]
    return handles


def draw_pr(ax, cohort, models, letter):
    y = Y[cohort]
    prev = float(np.mean(y))
    order = sorted(models, key=lambda m: average_precision(y, D[PFX[cohort] + m]),
                   reverse=True)
    for m in order:
        pr, rc, _ = precision_recall_curve(y, D[PFX[cohort] + m])
        ax.plot(rc, pr, color=COLORS[m], lw=LW, ls="-",
                solid_capstyle="round", zorder=2)
    ax.axhline(prev, color="#9A9A9A", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.045)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"{letter}   {NTITLE[cohort]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    handles = [Line2D([], [], color="#9A9A9A", lw=0.8, ls=(0, (4, 3)),
                      label=f"No-skill (prevalence {prev:.1%})")]
    handles += [Line2D([], [], color=COLORS[m], lw=LW,
                       label=f"{FULL[m]}  ({average_precision(y, D[PFX[cohort] + m]):.3f})")
                for m in order]
    return handles


def legend_below(ax, handles, ncol=2):
    return ax.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.0, -0.115), ncol=ncol,
                     borderaxespad=0.0, handlelength=1.6, handletextpad=0.4,
                     columnspacing=1.1, labelspacing=0.28, frameon=False,
                     fontsize=6.9)


# ----------------------------------------------------------------------
# Fig 3 — 2x2: A train ROC | B train PR | C test ROC | D test PR
# ----------------------------------------------------------------------
FIG_W, FIG_H = 11.8, 10.6
fig = plt.figure(figsize=(FIG_W, FIG_H))
AXES = {
    "A": fig.add_axes([0.062, 0.640, 0.372, 0.310]),
    "B": fig.add_axes([0.560, 0.640, 0.372, 0.310]),
    "C": fig.add_axes([0.062, 0.105, 0.372, 0.310]),
    "D": fig.add_axes([0.560, 0.105, 0.372, 0.310]),
}
legend_below(AXES["A"], draw_roc(AXES["A"], "train", NINE, "A"))
legend_below(AXES["B"], draw_pr(AXES["B"], "train", NINE, "B"))
legend_below(AXES["C"], draw_roc(AXES["C"], "test", NINE, "C"))
legend_below(AXES["D"], draw_pr(AXES["D"], "test", NINE, "D"))
fig.text(0.045, 0.985,
         "Nine candidate machine-learning models \u2014 ROC and precision-"
         "recall, training set (OOF) vs internal test set",
         ha="left", va="top", fontsize=10.2, fontweight="bold")
save_fig(fig, "Fig3_NineModels_TrainTest")
plt.close(fig)
print("[SAVED] Fig3_NineModels_TrainTest")

# ----------------------------------------------------------------------
# Fig S1 group — A train OOF ROC | B train OOF PR
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12.4, 6.0))
axA = fig.add_axes([0.058, 0.055, 0.372, 0.760])
axB = fig.add_axes([0.556, 0.055, 0.372, 0.760])
legend_below(axA, draw_roc(axA, "train", NINE, "A"))
legend_below(axB, draw_pr(axB, "train", NINE, "B"))
fig.text(0.04, 0.985,
         "Training-set out-of-fold curves \u2014 nine candidate models "
         "", ha="left", va="top", fontsize=10.2,
         fontweight="bold")
save_fig(fig, "FigS1_TrainOOF_ROC_PR")
plt.close(fig)
print("[SAVED] FigS1_TrainOOF_ROC_PR")

# ----------------------------------------------------------------------
# singles for the two slide-11 slots
# ----------------------------------------------------------------------
for tag, fn, letter in (("ROC", draw_roc, "A"), ("PR", draw_pr, "B")):
    fig = plt.figure(figsize=(6.4, 6.6))
    ax = fig.add_axes([0.135, 0.055, 0.795, 0.780])
    handles = fn(ax, "train", NINE, letter)
    legend_below(ax, handles, ncol=1)
    save_fig(fig, f"FigS1S_TrainOOF_{tag}")
    plt.close(fig)
    print(f"[SAVED] FigS1S_TrainOOF_{tag}")

# ----------------------------------------------------------------------
# Fig 4 C/G replacement — wide 2-panel PR, stacking + top-3
# ----------------------------------------------------------------------
TOP3 = [str(x) for x in D["top3"]]
MODELS4 = ["Stacking"] + TOP3
LW4 = {"Stacking": 2.0, "MLP": 1.3, "XGBoost": 1.3, "RF": 1.3}
COLORS4 = dict(COLORS)

fig = plt.figure(figsize=(12.6, 5.6))
axA = fig.add_axes([0.058, 0.045, 0.372, 0.800])
axB = fig.add_axes([0.556, 0.045, 0.372, 0.800])
for ax, coh, letter in ((axA, "test", "C"), (axB, "eICU", "G")):
    y = Y[coh]
    prev = float(np.mean(y))
    for m in MODELS4:
        pr, rc, _ = precision_recall_curve(y, D[PFX[coh] + m])
        ax.plot(rc, pr, color=COLORS4[m], lw=LW4[m], ls="-",
                solid_capstyle="round", zorder=3 if m == "Stacking" else 2)
    ax.axhline(prev, color="#9A9A9A", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.045)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"{letter}   {NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    hs = [Line2D([], [], color="#9A9A9A", lw=0.8, ls=(0, (4, 3)),
                 label=f"No-skill (prevalence {prev:.1%})")]
    for m in MODELS4:
        ap = average_precision(y, D[PFX[coh] + m])
        nm = "Stacking ensemble" if m == "Stacking" else FULL[m]
        hs.append(Line2D([], [], color=COLORS4[m], lw=LW4[m],
                         label=f"{nm}  (AP {ap:.3f})"))
    legend_below(ax, hs, ncol=2)
fig.text(0.04, 0.985,
         "Precision-recall \u2014 stacking ensemble vs the three selected "
         "base learners", ha="left", va="top",
         fontsize=10.2, fontweight="bold")
save_fig(fig, "Fig4CG_PR_two_cohorts")
plt.close(fig)
print("[SAVED] Fig4CG_PR_two_cohorts")
print("[DONE]")
