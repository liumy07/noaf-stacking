# -*- coding: utf-8 -*-
"""
step45_v80_flowcharts.py — study flowchart + stacking workflow, v80 numbers
(fold-internal selection, frozen predictor).
Output: v80_foldwise/figures/Figure_Flowchart_study.*, Figure_Workflow_stacking.*
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import set_style, save_fig  # noqa: E402

INK = "#1A2230"
DEEP = "#0E3F8C"
MAIN = "#1E4FA8"
SOFT = "#E8EFF8"
MIST = "#F7F9FC"
ACC = "#C0392B"
ACC_BG = "#FBF0EE"
GREY = "#8B97A8"

# ---- v80 headline numbers (from step40/step41/step44) -----------------------
TOP3_STR = "MLP 0.837   ·   RF 0.836   ·   XGBoost 0.833"
META_STR = "(w: MLP 2.47 · RF 2.21 · XGB 0.81)"
AUC_INT = "AUC 0.843\n(0.835–0.851)"
AUC_EXT = "AUC 0.753\n(0.743–0.762)"
AUC_INT_1L = "Internal test — AUC 0.843 (0.835–0.851)"
AUC_EXT_1L = "eICU external — AUC 0.753 (0.743–0.762)"
STAB_SUB = "10 explain sets · Spearman {RHO}"


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=1.4, z=2):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(b)
    return b


def txt(ax, x, y, s, size=8.6, color=INK, weight="normal", ha="center",
        va="center", style="normal", z=3, lsp=1.25, rotation=0):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, style=style, zorder=z, linespacing=lsp,
            rotation=rotation)


def arrow(ax, x1, y1, x2, y2, color=MAIN, lw=1.4, ms=9.0, style="-|>",
          con="arc3,rad=0"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=ms, color=color, lw=lw, zorder=3,
                        connectionstyle=con, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def study_flowchart():
    fig, ax = plt.subplots(figsize=(11.4, 6.6), dpi=300)
    ax.set_xlim(0, 114)
    ax.set_ylim(0, 66)
    ax.axis("off")

    txt(ax, 57, 63.4, "Study flowchart for atrial fibrillation prediction "
        "modelling", size=14.5, weight="bold", color=DEEP)

    for x, w, label in [(3, 31, "Phase 1 · Data curation"),
                        (40, 34, "Phase 2 · Modelling & validation"),
                        (80, 31, "Phase 3 · Interpretability")]:
        box(ax, x, 56.2, w, 4.2, DEEP, DEEP, r=1.2)
        txt(ax, x + w / 2, 58.3, label, size=10, color="white", weight="bold")

    # ---- phase 1 -------------------------------------------------------
    box(ax, 9, 49.6, 23, 4.0, SOFT, MAIN)
    txt(ax, 20.5, 51.6, "MIMIC-IV database", size=9.6, weight="bold",
        color=DEEP)
    box(ax, 3, 33.4, 35, 13.6, MIST, "#C9D4E4")
    txt(ax, 20.5, 45.2, "Inclusion criteria", size=9.2, weight="bold",
        color=DEEP)
    crit = ("Age > 18 years   ·   ICU stay > 2 days\n"
            "No atrial fibrillation on day 1\n"
            "No prior cardiac surgery / AF history\n"
            "First ICU admission only")
    txt(ax, 20.5, 39.6, crit, size=7.8, color=INK, lsp=1.5)
    box(ax, 7, 26.6, 27, 4.0, SOFT, MAIN)
    txt(ax, 20.5, 28.6, "Cohort selection — 34,384 patients", size=9.0,
        weight="bold", color=DEEP)
    box(ax, 3.5, 20.4, 15.5, 4.2, "#EAF3EC", "#2E8B57")
    txt(ax, 11.2, 22.5, "No NOAF\n24,849", size=8.4, color="#1E5C38",
        weight="bold")
    box(ax, 22.5, 20.4, 15.5, 4.2, ACC_BG, ACC)
    txt(ax, 30.2, 22.5, "NOAF\n9,535", size=8.4, color=ACC, weight="bold")
    box(ax, 3.5, 12.6, 15.5, 4.6, SOFT, MAIN)
    txt(ax, 11.2, 14.9, "Training 70%\n24,068", size=8.2, weight="bold",
        color=DEEP)
    box(ax, 22.5, 12.6, 15.5, 4.6, "#FFFFFF", MAIN)
    txt(ax, 30.2, 14.9, "Internal test 30%\n10,316  (locked)", size=8.2,
        weight="bold", color=DEEP)
    txt(ax, 20.5, 9.4, "Stratified 70/30 split · seed 42", size=7.6,
        color=GREY, style="italic")
    # straight arrows: start 0.3 outside the source box, stop 0.35 short of
    # the target box; all vertical runs share one axis (x=20.5) so the
    # left-column arrows are perfectly aligned
    arrow(ax, 20.5, 49.2, 20.5, 47.85)     # MIMIC-IV -> inclusion criteria
    arrow(ax, 20.5, 33.1, 20.5, 31.1)      # inclusion -> cohort selection
    arrow(ax, 14.2, 26.3, 12.2, 25.1)      # cohort -> No NOAF
    arrow(ax, 26.8, 26.3, 28.8, 25.1)      # cohort -> NOAF
    arrow(ax, 11.2, 20.1, 11.2, 17.7)      # No NOAF -> training 70%
    arrow(ax, 30.2, 20.1, 30.2, 17.7)      # NOAF -> internal test 30%

    # ---- phase 2 -------------------------------------------------------
    box(ax, 43, 49.6, 28, 4.0, SOFT, MAIN)
    txt(ax, 57, 51.6, "39 harmonised variables", size=9.4, weight="bold",
        color=DEEP)
    box(ax, 42.5, 44.4, 29, 3.8, MIST, "#C9D4E4")
    txt(ax, 57, 46.3, "Univariable screening — 37 / 39 significant",
        size=8.4)
    box(ax, 42.5, 38.2, 29, 4.0, MIST, "#C9D4E4")
    txt(ax, 57, 40.2, "LASSO — scikit-learn · 10-fold CV · 1-SE rule",
        size=8.4)
    box(ax, 42.5, 29.8, 29, 5.4, MIST, "#C9D4E4")
    txt(ax, 57, 32.5, "9 base learners on the 26 retained variables ·\n"
        "10-fold CV with fold-internal imputation\nand re-selection;\n"
        "test & eICU excluded from every decision",
        size=7.8)
    box(ax, 42.5, 21.3, 29, 6.6, ACC_BG, ACC, lw=1.5)
    txt(ax, 57, 25.9, "Top 3 by training-set OOF AUC", size=8.8,
        weight="bold", color=ACC)
    txt(ax, 57, 23.3, TOP3_STR, size=8.2, color=INK)
    box(ax, 42.5, 15.2, 29, 4.0, SOFT, MAIN)
    txt(ax, 57, 17.2, "LASSO meta-learner\n" + META_STR,
        size=7.8, weight="bold", color=DEEP)
    box(ax, 42.5, 8.4, 29, 4.4, MAIN, MAIN)
    txt(ax, 57, 10.6, "Stacking ensemble (frozen predictor)", size=9.0,
        color="white", weight="bold")
    box(ax, 42.5, 1.6, 13.5, 4.2, "#FFFFFF", MAIN)
    txt(ax, 49.2, 3.7, "Internal test\n" + AUC_INT, size=8.0, weight="bold",
        color=DEEP)
    box(ax, 58.0, 1.6, 13.5, 4.2, "#FFFFFF", MAIN)
    txt(ax, 64.8, 3.7, "eICU external\n" + AUC_EXT, size=8.0, weight="bold",
        color=DEEP)
    for y1, y2 in [(49.3, 47.9), (44.1, 42.55), (37.9, 35.55),
                   (29.5, 28.25), (21.0, 19.55), (14.9, 13.15)]:
        arrow(ax, 57, y1, 57, y2)
    arrow(ax, 49.2, 8.1, 49.2, 6.2)
    arrow(ax, 64.8, 8.1, 64.8, 6.2)
    # cohort data feeds the modelling column: straight run through the
    # corridor between the phase columns, tip pointing at the FIRST
    # phase-2 box ("39 harmonised variables"), stopping short of its edge
    arrow(ax, 38.4, 14.9, 42.4, 51.6, con="arc3,rad=0.03", lw=1.3)

    # ---- phase 3 -------------------------------------------------------
    p3 = [("SHAP value calculation\n(complete frozen stacking pipeline)",
           "Probability-scale attribution"),
          ("Global feature importance\n+ sampling stability",
           STAB_SUB),
          ("Patient-level explanation\nTP / TN / FP / FN waterfalls",
           "Individual decision profiles")]
    ytop = 51.6
    for i, (head, sub) in enumerate(p3):
        y = ytop - i * 12.4
        box(ax, 82, y - 3.6, 27, 6.4, SOFT if i % 2 == 0 else MIST, MAIN)
        txt(ax, 95.5, y + 0.7, head, size=8.4, weight="bold", color=DEEP)
        txt(ax, 95.5, y - 1.8, sub, size=7.4, color=INK)
        if i < 2:
            arrow(ax, 95.5, y - 3.8, 95.5, y - 8.6)
    arrow(ax, 71.9, 10.6, 81.6, 49.5, con="arc3,rad=-0.18", lw=1.3)

    # ---- bottom banner -------------------------------------------------
    for x, w, label in [(3, 31, "Data curation"),
                        (40, 34, "Modelling & validation"),
                        (80, 31, "Interpretability")]:
        box(ax, x, -3.4, w, 3.0, SOFT, "#C9D4E4", lw=0.8, r=0.8)
        txt(ax, x + w / 2, -1.9, label, size=8.2, weight="bold", color=MAIN)
    txt(ax, 111, -5.6, "NOAF = new-onset atrial fibrillation", size=7.4,
        color=GREY, ha="right", style="italic")

    fig.tight_layout()
    save_fig(fig, "Figure_Flowchart_study")
    plt.close(fig)


def stacking_workflow():
    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=300)
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 64)
    ax.axis("off")

    txt(ax, 48, 61.4, "Stacking ensemble model development workflow",
        size=13.5, weight="bold", color=DEEP)

    txt(ax, 10, 54.6, "Step 1", size=10.5, weight="bold", color=DEEP)
    txt(ax, 10, 51.9, "Leakage-free\nmodel selection", size=8.2, color=GREY)
    box(ax, 22, 47.4, 60, 9.2, SOFT, MAIN, lw=1.3)
    txt(ax, 52, 53.6, "9 candidate machine-learning models", size=10,
        weight="bold", color=DEEP)
    txt(ax, 52, 50.2, "ranked by training-set 10-fold OUT-OF-FOLD AUC, "
        "with imputation, univariable + LASSO re-run\ninside every fold; internal test "
        "set and eICU excluded from every development decision",
        size=7.8, color=INK, lsp=1.4)

    cards = [("MLP", "OOF AUC 0.837"), ("RF", "OOF AUC 0.836"),
             ("XGBoost", "OOF AUC 0.833")]
    xs = [24.5, 41.5, 58.5]
    for (name, sub), x in zip(cards, xs):
        box(ax, x, 35.4, 15.5, 7.2, "#FFFFFF", MAIN, lw=1.3)
        txt(ax, x + 7.75, 40.0, name, size=10, weight="bold", color=DEEP)
        txt(ax, x + 7.75, 37.4, "base learner\n" + sub, size=7.0, color=INK)
        arrow(ax, 52, 47.2, x + 7.75, 42.9, lw=1.1, ms=7)

    txt(ax, 80.5, 39.0, "6 candidates\nnot selected", size=7.6, color=GREY,
        style="italic")
    arrow(ax, 74.2, 39.0, 77.2, 39.0, color=GREY, lw=1.0, ms=6)

    txt(ax, 10, 27.4, "Step 2", size=10.5, weight="bold", color=DEEP)
    txt(ax, 10, 24.7, "Meta-model\ntraining", size=8.2, color=GREY)
    arrow(ax, 52, 35.2, 52, 30.2, lw=1.5)
    txt(ax, 62.5, 32.6, "out-of-fold NOAF probabilities", size=7.6,
        color=INK, ha="left")
    box(ax, 22, 21.4, 60, 8.4, "#D9E6F7", MAIN, lw=1.3)
    txt(ax, 52, 27.4, "LASSO logistic-regression meta-learner", size=10,
        weight="bold", color=DEEP)
    txt(ax, 52, 23.8, "combines the three base-model predictions\n"
        "weights: MLP +2.47 · RF +2.21 · XGBoost +0.81", size=7.8, color=INK,
        lsp=1.4)

    txt(ax, 10, 11.6, "Step 3", size=10.5, weight="bold", color=DEEP)
    txt(ax, 10, 8.9, "Stacking\nensemble", size=8.2, color=GREY)
    arrow(ax, 52, 21.2, 52, 16.6, lw=1.5)
    box(ax, 22, 8.0, 60, 8.4, MAIN, MAIN, lw=1.3)
    txt(ax, 52, 14.0, "Stacking ensemble model (frozen)", size=10.5,
        color="white", weight="bold")
    txt(ax, 52, 10.4, "MLP + RF + XGBoost combined via LASSO — one predictor "
        "for evaluation and SHAP", size=7.6, color="#DCE7F7")

    box(ax, 24, 0.6, 26, 4.6, "#FFFFFF", MAIN, lw=1.1)
    txt(ax, 37, 2.9, AUC_INT_1L, size=7.6, weight="bold", color=DEEP)
    box(ax, 54, 0.6, 26, 4.6, "#FFFFFF", MAIN, lw=1.1)
    txt(ax, 67, 2.9, AUC_EXT_1L, size=7.6, weight="bold", color=DEEP)
    arrow(ax, 40, 7.8, 37, 5.4, lw=1.1, ms=7)
    arrow(ax, 64, 7.8, 67, 5.4, lw=1.1, ms=7)

    fig.tight_layout()
    save_fig(fig, "Figure_Workflow_stacking")
    plt.close(fig)


if __name__ == "__main__":
    set_style()
    study_flowchart()
    stacking_workflow()
    print("[saved] Figure_Flowchart_study + Figure_Workflow_stacking (v80)")
