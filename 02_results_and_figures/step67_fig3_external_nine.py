# -*- coding: utf-8 -*-
"""
step67_fig3_external_nine.py

WHY
---
Section 3.3 reports the nine candidate learners in three settings - training
set (OOF), internal hold-out test set and the external eICU-CRD cohort - but
Figure 3 only carried the first two; the external numbers existed as text and
Table 2 only.  This script adds that missing visual layer.

OUTPUT
------
1. Fig3_NineModels_AllCohorts.png/.pdf/.svg
   Figure 3 rebuilt as a 3 x 2 panel matrix:
       A/B  training set, 10-fold out-of-fold (n = 24,068)
       C/D  internal hold-out test set        (n = 10,316)
       E/F  external eICU-CRD cohort          (n = 32,067)
   Existing panels A-D keep the identical style, colours, model order,
   axis limits and legend format, so only the new bottom row is new.

2. Fig3EF_External_ROC_PR.png/.pdf/.svg
   The bottom row alone, full-width, for the deck slide.

No statistic is recomputed differently: curves come from the frozen
predictions in v80_scores.npz and every AUC/AP in the legends is read from
v80_nine_models_metrics.csv / v80_ci_ap.csv, exactly as in step43.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "00_common"))

from noaf_common_v80 import (  # noqa: E402
    V80_DIR, FIG_DIR, COLORS, set_style, save_fig, fast_auc, average_precision,
)

plt = set_style()
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import roc_curve, precision_recall_curve  # noqa: E402

D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
NINE = ["MLP", "XGBoost", "RF", "LightGBM", "Logistic", "SVM", "Ridge",
        "DT", "KNN"]
FULL = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
        "RF": "Random forest", "LightGBM": "LightGBM",
        "Logistic": "Logistic regression", "SVM": "Support vector machine",
        "Ridge": "Ridge regression", "DT": "Decision tree",
        "KNN": "K-nearest neighbours", "Stacking": "Stacking ensemble"}
Y = {"train": D["y_train"].astype(int), "test": D["y_test"].astype(int),
     "eICU": D["y_eicu"].astype(int)}
PFX = {"train": "oof__", "test": "tst__", "eICU": "ext__"}
NTITLE = {
    "train": "Training set \u2014 10-fold out-of-fold (n = 24,068)",
    "test": "Internal hold-out test set (n = 10,316)",
    "eICU": "External eICU-CRD cohort (n = 32,067)"}

AUC_CI = pd.read_csv(V80_DIR / "v80_nine_models_metrics.csv",
                     encoding="utf-8-sig").set_index(["Model", "cohort"])
AP_CI = pd.read_csv(V80_DIR / "v80_ci_ap.csv",
                    encoding="utf-8-sig").set_index(["Model", "cohort"])


def auc_ci(m, coh):
    # use the preformatted column (computed from the RAW CIs) so the legend
    # cannot drift from Table 2 through double rounding of the 4-dp columns
    return str(AUC_CI.loc[(m, coh), "AUC_CI"]).replace("-", "\u2013")


def ap_ci(m, coh):
    r = AP_CI.loc[(m, coh)]
    return f"AP {r.AP:.3f} ({r.lo:.3f}\u2013{r.hi:.3f})"


def legend_below(ax, handles, ncol=2, fs=6.4, y=-0.135):
    return ax.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.0, y), ncol=ncol,
                     borderaxespad=0.0, handlelength=1.5, handletextpad=0.4,
                     columnspacing=0.9, labelspacing=0.22, frameon=False,
                     fontsize=fs)


def roc_draw(ax, coh, models, letter):
    y = Y[coh]
    order = sorted(models, key=lambda m: fast_auc(y, D[PFX[coh] + m]),
                   reverse=True)
    for m in order:
        fpr, tpr, _ = roc_curve(y, D[PFX[coh] + m])
        ax.plot(fpr, tpr, color=COLORS[m], lw=1.25, ls="-",
                solid_capstyle="round", zorder=2)
    ax.plot([0, 1], [0, 1], color="#B0B0B0", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_xlabel("1 \u2212 specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{letter}   {NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    hs = [Line2D([], [], color="#B0B0B0", lw=0.8, ls=(0, (4, 3)),
                 label="Chance (AUC 0.500)")]
    for m in order:
        hs.append(Line2D([], [], color=COLORS[m], lw=1.25,
                         label=f"{FULL[m]}  ({auc_ci(m, coh)})"))
    return hs


def pr_draw(ax, coh, models, letter):
    y = Y[coh]
    prev = float(np.mean(y))
    order = sorted(models, key=lambda m: average_precision(y, D[PFX[coh] + m]),
                   reverse=True)
    for m in order:
        pr, rc, _ = precision_recall_curve(y, D[PFX[coh] + m])
        ax.plot(rc, pr, color=COLORS[m], lw=1.25, ls="-",
                solid_capstyle="round", zorder=2)
    ax.axhline(prev, color="#9A9A9A", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.045)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"{letter}   {NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    hs = [Line2D([], [], color="#9A9A9A", lw=0.8, ls=(0, (4, 3)),
                 label=f"No-skill (prevalence {prev:.1%})")]
    for m in order:
        hs.append(Line2D([], [], color=COLORS[m], lw=1.25,
                         label=f"{FULL[m]}  ({ap_ci(m, coh)})"))
    return hs


# ----------------------------------------------------------------------
# sanity check: the curves must reproduce Table 2 exactly
# ----------------------------------------------------------------------
print("[CHECK] external AUC from npz vs metrics table")
for m in NINE:
    a = fast_auc(Y["eICU"], D["ext__" + m])
    b = float(AUC_CI.loc[(m, "eICU"), "AUC"])
    flag = "OK " if abs(a - b) < 5e-4 else "DIFF"
    print(f"  {flag} {m:<9} npz {a:.4f}  table {b:.4f}")

# ----------------------------------------------------------------------
# 1. Figure 3, 3 x 2: OOF / internal test / external
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12.6, 15.2))
AX = {}
for key, (xa, ya) in {
        "A": (0.062, 0.716), "B": (0.560, 0.716),
        "C": (0.062, 0.404), "D": (0.560, 0.404),
        "E": (0.062, 0.092), "F": (0.560, 0.092)}.items():
    AX[key] = fig.add_axes([xa, ya, 0.372, 0.212])

legend_below(AX["A"], roc_draw(AX["A"], "train", NINE, "A"))
legend_below(AX["B"], pr_draw(AX["B"], "train", NINE, "B"))
legend_below(AX["C"], roc_draw(AX["C"], "test", NINE, "C"))
legend_below(AX["D"], pr_draw(AX["D"], "test", NINE, "D"))
legend_below(AX["E"], roc_draw(AX["E"], "eICU", NINE, "E"))
legend_below(AX["F"], pr_draw(AX["F"], "eICU", NINE, "F"))

fig.text(0.045, 0.990,
         "Nine candidate machine-learning models \u2014 ROC and "
         "precision-recall curves in the training set (out-of-fold), the "
         "internal test set and the external cohort",
         ha="left", va="top", fontsize=10.6, fontweight="bold")
save_fig(fig, "Fig3_NineModels_AllCohorts")
plt.close(fig)
print("[SAVED] Fig3_NineModels_AllCohorts")

# ----------------------------------------------------------------------
# 2. the external row alone, for the deck
# ----------------------------------------------------------------------
fig = plt.figure(figsize=(12.6, 6.5))
axE = fig.add_axes([0.058, 0.085, 0.372, 0.735])
axF = fig.add_axes([0.556, 0.085, 0.372, 0.735])
legend_below(axE, roc_draw(axE, "eICU", NINE, "E"), fs=6.6, y=-0.125)
legend_below(axF, pr_draw(axF, "eICU", NINE, "F"), fs=6.6, y=-0.125)
fig.text(0.040, 0.990,
         "External validation (eICU-CRD, n = 32,067) \u2014 ROC and "
         "precision-recall curves of the nine candidate models",
         ha="left", va="top", fontsize=10.6, fontweight="bold")
save_fig(fig, "Fig3EF_External_ROC_PR")
plt.close(fig)
print("[SAVED] Fig3EF_External_ROC_PR")

print("[DIR]", FIG_DIR)
