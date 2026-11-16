# -*- coding: utf-8 -*-
"""
step30_v80_figures_ci.py — regenerate EVERY result figure of the deck from the
v70 leak-free pipeline, with a 95% CI attached to every annotated number
(AUC: DeLong; AP / net benefit / calibration: stratified bootstrap 1,000).

All legends stay outside the axes; every in-axes box is placed by the
collision-free locator (place_block).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import (  # noqa: E402
    V80_DIR, FIG_DIR, COLORS, set_style, save_fig, fast_auc,
    average_precision, net_benefit, LABELS,
)

plt = set_style()
import matplotlib as mpl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import roc_curve, precision_recall_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
TOP3 = [str(x) for x in D["top3"]]
NINE = ["MLP", "XGBoost", "RF", "LightGBM", "Logistic", "SVM", "Ridge",
        "DT", "KNN"]
FULL = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
        "RF": "Random forest", "LightGBM": "LightGBM",
        "Logistic": "Logistic regression", "SVM": "Support vector machine",
        "Ridge": "Ridge regression", "DT": "Decision tree",
        "KNN": "K-nearest neighbours", "Stacking": "Stacking ensemble"}
SHORT = {"Stacking": "Stacking", "MLP": "MLP", "XGBoost": "XGBoost",
         "RF": "RF", "LightGBM": "LightGBM"}
LW = {"Stacking": 2.0}
Y = {"train": D["y_train"].astype(int), "test": D["y_test"].astype(int),
     "eICU": D["y_eicu"].astype(int)}
PFX = {"train": "oof__", "test": "tst__", "eICU": "ext__"}
NTITLE = {
    "train": "Training set — 10-fold out-of-fold (n = 24,068)",
    "test": "Internal hold-out test set (n = 10,316)",
    "eICU": "eICU-CRD external cohort (n = 32,067)"}

# ---- CI tables ----------------------------------------------------------
AUC_CI = pd.read_csv(V80_DIR / "v80_nine_models_metrics.csv",
                     encoding="utf-8-sig").set_index(["Model", "cohort"])
AP_CI = pd.read_csv(V80_DIR / "v80_ci_ap.csv",
                    encoding="utf-8-sig").set_index(["Model", "cohort"])
NB_CI = pd.read_csv(V80_DIR / "v80_ci_nb.csv", encoding="utf-8-sig")
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv",
                  encoding="utf-8-sig").set_index("cohort")
DL = pd.read_csv(V80_DIR / "v80_delong_paired.csv", encoding="utf-8-sig")
LASSO_PATH = pd.read_csv(V80_DIR / "v80_lasso_path.csv", encoding="utf-8-sig")
LASSO_COEF = pd.read_csv(V80_DIR / "v80_lasso_path_coef.csv",
                         encoding="utf-8-sig")
UNI = pd.read_csv(V80_DIR / "v80_univariable_logistic.csv",
                  encoding="utf-8-sig")
UNI["label"] = UNI["var"].map(LABELS)
_sel_txt = (V80_DIR / "v80_lasso_selected.txt").read_text(encoding="utf-8")
N_SEL = int(re.search(r"LASSO retained:\s*(\d+)", _sel_txt).group(1))
N_UNI = int(re.search(r"Univariable input:\s*(\d+)", _sel_txt).group(1))


def auc_ci(m, coh):
    r = AUC_CI.loc[(m, coh)]
    return f"{r.AUC:.3f} ({r.CI_lo:.3f}\u2013{r.CI_hi:.3f})"


def ap_ci(m, coh):
    r = AP_CI.loc[(m, coh)]
    return f"AP {r.AP:.3f} ({r.lo:.3f}\u2013{r.hi:.3f})"


# ---- shared drawing helpers ---------------------------------------------
def place_block(ax, text, curves, fs=6.2, name="block"):
    fig = ax.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    t = ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=fs, zorder=6, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor="#DDDDDD", linewidth=0.4, alpha=0.96))
    fig.canvas.draw()
    bb0 = t.get_window_extent(ren)
    w, h = bb0.width, bb0.height
    pts_all = np.vstack([ax.transData.transform(np.column_stack([xs, ys]))
                         for xs, ys in curves.values()])
    ax_bb = ax.get_window_extent(ren)
    best = None
    for ya in np.arange(0.02, 1.001, 0.02):
        for xa in np.arange(1.0, 0.499, -0.02):
            x1, y0 = ax_bb.x0 + xa * ax_bb.width, ax_bb.y0 + ya * ax_bb.height
            x0, y1 = x1 - w, y0 + h
            if x0 < ax_bb.x0 or y1 > ax_bb.y1:
                continue
            m = 3.0
            hit = ((pts_all[:, 0] > x0 - m) & (pts_all[:, 0] < x1 + m) &
                   (pts_all[:, 1] > y0 - m) & (pts_all[:, 1] < y1 + m)).sum()
            if hit == 0:
                best = (xa, ya)
                break
        if best:
            break
    if best is None:
        t.remove()
        raise RuntimeError(f"[{name}] no collision-free position")
    t.set_position(best)
    return t


def legend_below(ax, handles, ncol=1, fs=6.9):
    return ax.legend(handles=handles, loc="upper left",
                     bbox_to_anchor=(0.0, -0.115), ncol=ncol,
                     borderaxespad=0.0, handlelength=1.6, handletextpad=0.4,
                     columnspacing=1.0, labelspacing=0.28, frameon=False,
                     fontsize=fs)


def roc_draw(ax, coh, models, letter, title=None):
    y = Y[coh]
    order = sorted(models, key=lambda m: fast_auc(y, D[PFX[coh] + m]),
                   reverse=True)
    for m in order:
        fpr, tpr, _ = roc_curve(y, D[PFX[coh] + m])
        ax.plot(fpr, tpr, color=COLORS[m],
                lw=LW.get(m, 1.25), ls="-", solid_capstyle="round",
                zorder=3 if m == "Stacking" else 2)
    ax.plot([0, 1], [0, 1], color="#B0B0B0", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.015)
    ax.set_xlabel("1 \u2212 specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{letter + '   ' if letter else ''}{title or NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    hs = [Line2D([], [], color="#B0B0B0", lw=0.8, ls=(0, (4, 3)),
                 label="Chance (AUC 0.500)")]
    for m in order:
        hs.append(Line2D([], [], color=COLORS[m], lw=LW.get(m, 1.25),
                         label=f"{FULL[m]}  ({auc_ci(m, coh)})"))
    return hs


def pr_draw(ax, coh, models, letter, title=None):
    y = Y[coh]
    prev = float(np.mean(y))
    order = sorted(models, key=lambda m: average_precision(y, D[PFX[coh] + m]),
                   reverse=True)
    for m in order:
        pr, rc, _ = precision_recall_curve(y, D[PFX[coh] + m])
        ax.plot(rc, pr, color=COLORS[m], lw=LW.get(m, 1.25), ls="-",
                solid_capstyle="round", zorder=3 if m == "Stacking" else 2)
    ax.axhline(prev, color="#9A9A9A", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.015, 1.045)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(f"{letter + '   ' if letter else ''}{title or NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    hs = [Line2D([], [], color="#9A9A9A", lw=0.8, ls=(0, (4, 3)),
                 label=f"No-skill (prevalence {prev:.1%})")]
    for m in order:
        hs.append(Line2D([], [], color=COLORS[m], lw=LW.get(m, 1.25),
                         label=f"{FULL[m]}  ({ap_ci(m, coh)})"))
    return hs


def dca_draw(ax, coh, models, letter, with_box=True):
    y = Y[coh]
    prev = float(np.mean(y))
    thr = np.linspace(0.01, 0.50, 130)
    ax.axhline(0, color="#9A9A9A", lw=0.7, ls=(0, (3, 2)), zorder=1)
    curves, handles = {}, []
    handles.append(Line2D([], [], color="#9A9A9A", lw=0.7, ls=(0, (3, 2)),
                          label="Treat none (net benefit = 0)"))
    curves["none"] = (thr, np.zeros_like(thr))
    for m in models:
        nb = net_benefit(y, D[PFX[coh] + m], thr)
        ax.plot(thr, nb, color=COLORS[m], lw=LW.get(m, 1.25), ls="-",
                solid_capstyle="round", zorder=3 if m == "Stacking" else 2)
        curves[m] = (thr, nb)
        handles.append(Line2D([], [], color=COLORS[m], lw=LW.get(m, 1.25),
                              label=f"{FULL[m]}  ({auc_ci(m, coh)})"))
    ax.set_xlim(0.01, 0.50)
    lo = min(0.0, float(min(np.min(v[1]) for v in curves.values())))
    hi = max(float(max(np.max(v[1]) for v in curves.values())), prev)
    span = hi - lo
    ax.set_ylim(lo - 0.04 * span, hi + 0.08 * span)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"{letter}   {NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    art = None
    if with_box:
        lines = ["Net benefit (95% bootstrap CI)",
                 "threshold 0.10 / 0.20"]
        for m in models:
            r10 = NB_CI[(NB_CI.Model == m) & (NB_CI.cohort == coh)
                        & (NB_CI.thr == 0.10)].iloc[0]
            r20 = NB_CI[(NB_CI.Model == m) & (NB_CI.cohort == coh)
                        & (NB_CI.thr == 0.20)].iloc[0]
            lines.append(f"{SHORT[m]}  {r10.nb:+.3f} ({r10.lo:+.3f}, "
                         f"{r10.hi:+.3f})  /  {r20.nb:+.3f} "
                         f"({r20.lo:+.3f}, {r20.hi:+.3f})")
        art = place_block(ax, "\n".join(lines), curves, fs=5.9,
                          name=f"DCA box {coh}")
    return handles, art


def calib_draw(ax, coh, letter):
    y, s = Y[coh], D[PFX[coh] + "Stacking"]
    eps = 1e-6
    lp = np.log(np.clip(s, eps, 1 - eps) / (1 - np.clip(s, eps, 1 - eps)))
    lr = LogisticRegression(C=1e6, max_iter=500).fit(lp.reshape(-1, 1), y)
    p_grid = np.linspace(0.01, 0.99, 10)
    obs = []
    for j in range(len(p_grid) - 1):
        m_ = (s >= p_grid[j]) & (s < p_grid[j + 1])
        if m_.sum() < 30:
            obs.append((np.nan, np.nan, 0))
            continue
        obs.append((float(np.mean(lp[m_] * 0 + s[m_])),
                    float(np.mean(y[m_])), int(m_.sum())))
    # deciles of predicted probability
    qs = np.quantile(s, np.linspace(0, 1, 11))
    xs, ys_, ns_ = [], [], []
    rng = np.random.default_rng(42)
    for j in range(10):
        m_ = (s >= qs[j]) & (s < qs[j + 1] if j < 9 else s <= qs[j + 1])
        if m_.sum() < 30:
            continue
        xs.append(float(np.mean(s[m_])))
        ys_.append(float(np.mean(y[m_])))
        ns_.append(int(m_.sum()))
    # bootstrap CI band around the calibrated curve
    boots = []
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    grid = np.linspace(0.005, 0.995, 60)
    for _ in range(400):
        ix = np.concatenate([rng.choice(pos, len(pos), True),
                             rng.choice(neg, len(neg), True)])
        lpb = lp[ix]
        lrb = LogisticRegression(C=1e6, max_iter=500).fit(
            lpb.reshape(-1, 1), y[ix])
        boots.append(lrb.predict_proba(
            np.log(grid / (1 - grid)).reshape(-1, 1))[:, 1])
    boots = np.asarray(boots)
    blo, bhi = np.percentile(boots, [2.5, 97.5], axis=0)
    ax.fill_between(grid, blo, bhi, color="#D64545", alpha=0.18,
                    lw=0, zorder=2)
    ax.plot(grid, np.median(boots, axis=0), color="#B2182B", lw=1.6,
            zorder=3, label="Stacking ensemble (calibrated)")
    ax.plot([0, 1], [0, 1], color="#808080", lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted probability of NOAF")
    ax.set_ylabel("Observed proportion of NOAF")
    ax.set_title(f"{letter}   {NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    r = CAL.loc[coh]
    txt = (f"Intercept {r['intercept']:+.3f} ({r['i_lo']:+.3f} to {r['i_hi']:+.3f})\n"
           f"Slope {r['slope']:.3f} ({r['s_lo']:.3f} to {r['s_hi']:.3f})\n"
           f"Brier {r.brier:.4f}")
    art = place_block(ax, txt, {"cal": (grid, np.median(boots, axis=0)),
                                "ideal": ([0, 1], [0, 1])},
                      fs=6.6, name=f"cal {coh}")
    hs = [Line2D([], [], color="#808080", lw=0.9, ls=(0, (5, 3)),
                 label="Ideal (slope 1, intercept 0)"),
          Line2D([], [], color="#B2182B", lw=1.6,
                 label="Stacking ensemble (calibrated)")]
    return hs, art


# ==========================================================================
# 1. Figure 2 — screening (A forest, B LASSO paths, C 1-SE rule)
# ==========================================================================
fig = plt.figure(figsize=(13.0, 9.6))
axA = fig.add_axes([0.335, 0.055, 0.175, 0.875])
axB = fig.add_axes([0.585, 0.565, 0.335, 0.345])
axC = fig.add_axes([0.585, 0.075, 0.335, 0.345])

u = UNI.sort_values("OR", ascending=False).reset_index(drop=True)
ylab = list(u["label"])[::-1]
ors = list(u["OR"])[::-1]
los = list(u["lo"])[::-1]
his = list(u["hi"])[::-1]
ps = list(u["p"])[::-1]
yy = np.arange(len(ors))
for i in range(len(ors)):
    col = "#2166AC" if ps[i] < 0.05 else "#B8B8B8"
    axA.plot([los[i], his[i]], [yy[i], yy[i]], color=col, lw=1.0, zorder=2)
    axA.plot(ors[i], yy[i], "o", color=col, ms=3.0, zorder=3)
axA.axvline(1, color="#888888", lw=0.8, ls=(0, (4, 3)), zorder=1)
axA.set_yticks(yy)
axA.set_yticklabels(ylab, fontsize=6.8)
axA.set_xscale("log")
axA.set_xlim(0.35, 6.5)
axA.minorticks_off()
axA.set_xticks([0.5, 1, 2, 3, 4, 5])
axA.set_xticklabels(["0.5", "1", "2", "3", "4", "5"])
axA.set_xlabel("Odds ratio for NOAF (log scale, 95% CI)")
axA.set_title("A   Univariable logistic regression (39 candidates)",
              loc="left", fontsize=9.0, fontweight="bold", pad=5)
axA.text(0.03, 0.985,
         f"P < 0.05 (blue): {int((u['p'] < 0.05).sum())} / 39   |   "
         f"MIMIC-IV training set (n = 24,068)",
         transform=axA.transAxes, ha="left", va="top", fontsize=6.6,
         style="italic", color="#333333")

pc = LASSO_COEF
lam = pc["lambda_"].values
feat_cols = [c for c in pc.columns if c not in ("step", "C", "lambda_")]
mag = pc[feat_cols].abs()
top = mag.max().sort_values(ascending=False)
top8 = list(top.index[:8])
palette = ["#2166AC", "#B2182B", "#1B7837", "#E08214", "#8073AC",
           "#01665E", "#8C510A", "#666666"]
for i, c in enumerate(feat_cols):
    if c in top8:
        axB.plot(np.log10(lam), mag[c], color=palette[top8.index(c)],
                 lw=1.3, zorder=3)
    else:
        axB.plot(np.log10(lam), mag[c], color="#CCCCCC", lw=0.6, zorder=2)
axB.set_xlabel("log$_{10}$(lambda)")
axB.set_ylabel("Coefficient magnitude (group max)")
axB.set_title("B   LASSO coefficient paths (L1, 10-fold CV grid)",
              loc="left", fontsize=9.0, fontweight="bold", pad=5)
hsB = [Line2D([], [], color=palette[i], lw=1.3, label=FULL.get(c, c))
       for i, c in enumerate(top8)]
axB.legend(handles=hsB, loc="upper left", bbox_to_anchor=(0.0, -0.175),
           ncol=2, frameon=False, fontsize=6.4, handlelength=1.4,
           handletextpad=0.4, columnspacing=0.9, labelspacing=0.25)

cv = LASSO_PATH
lam_cv = cv["lambda_"].values
m_ = cv["cv_auc"].values
se_ = cv["se"].values
best = int(np.argmax(m_))
thr_ = m_[best] - se_[best]
ok = np.where(m_ >= thr_)[0]
i1se = int(ok[np.argmax(lam_cv[ok])])
axC.fill_between(np.log10(lam_cv), m_ - se_, m_ + se_, color="#9BB8D8",
                 alpha=0.45, lw=0)
axC.plot(np.log10(lam_cv), m_, color="#2166AC", lw=1.4)
axC.axhline(thr_, color="#B8B8B8", lw=0.8, ls=(0, (4, 3)))
axC.axvline(np.log10(lam_cv[best]), color="#9A9A9A", lw=0.8,
            ls=(0, (4, 3)))
axC.axvline(np.log10(lam_cv[i1se]), color="#B2182B", lw=1.0, ls=(0, (5, 3)))
axC.plot(np.log10(lam_cv[i1se]), m_[i1se], "o", color="#B2182B", ms=4.5,
         zorder=4)
axC.set_xlabel("log$_{10}$(lambda)")
axC.set_ylabel("10-fold CV AUC (mean \u00b1 SE)")
axC.set_title("C   Cross-validated AUC and the 1-SE rule",
              loc="left", fontsize=9.0, fontweight="bold", pad=5)
axC.text(0.97, 0.06,
         f"1-SE rule: {N_SEL} variables\n"
         f"CV-AUC {m_[i1se]:.3f} at lambda = {lam_cv[i1se]:.3g}",
         transform=axC.transAxes, ha="right", va="bottom", fontsize=6.8,
         color="#B2182B")
fig.suptitle("Screening and selection of candidate predictors "
             "", x=0.045, y=0.985, ha="left",
             fontsize=10.4, fontweight="bold")
save_fig(fig, "Fig2_Screening")
plt.close(fig)
print("[SAVED] Fig2_Screening")

# ==========================================================================
# 2. Figure 3 — nine models, train OOF vs internal test (CI in legends)
# ==========================================================================
fig = plt.figure(figsize=(11.8, 10.6))
AX = {"A": fig.add_axes([0.062, 0.640, 0.372, 0.310]),
      "B": fig.add_axes([0.560, 0.640, 0.372, 0.310]),
      "C": fig.add_axes([0.062, 0.105, 0.372, 0.310]),
      "D": fig.add_axes([0.560, 0.105, 0.372, 0.310])}
legend_below(AX["A"], roc_draw(AX["A"], "train", NINE, "A"), ncol=2, fs=6.4)
legend_below(AX["B"], pr_draw(AX["B"], "train", NINE, "B"), ncol=2, fs=6.4)
legend_below(AX["C"], roc_draw(AX["C"], "test", NINE, "C"), ncol=2, fs=6.4)
legend_below(AX["D"], pr_draw(AX["D"], "test", NINE, "D"), ncol=2, fs=6.4)
fig.text(0.045, 0.988,
         "Nine candidate machine-learning models — ROC and precision-recall, "
         "training set (OOF) vs internal test set; 95% CI in the legends "
         "",
         ha="left", va="top", fontsize=10.0, fontweight="bold")
save_fig(fig, "Fig3_NineModels_TrainTest")
plt.close(fig)
print("[SAVED] Fig3_NineModels_TrainTest")

# ==========================================================================
# 3. Fig S1 group + singles
# ==========================================================================
fig = plt.figure(figsize=(12.4, 6.0))
axA = fig.add_axes([0.058, 0.055, 0.372, 0.760])
axB = fig.add_axes([0.556, 0.055, 0.372, 0.760])
legend_below(axA, roc_draw(axA, "train", NINE, "A"), ncol=2, fs=6.4)
legend_below(axB, pr_draw(axB, "train", NINE, "B"), ncol=2, fs=6.4)
fig.text(0.04, 0.988,
         "Training-set out-of-fold curves — nine candidate models "
         "", ha="left", va="top", fontsize=10.2,
         fontweight="bold")
save_fig(fig, "FigS1_TrainOOF_ROC_PR")
plt.close(fig)
print("[SAVED] FigS1_TrainOOF_ROC_PR")

for tag, fn, letter in (("ROC", roc_draw, "A"), ("PR", pr_draw, "B")):
    fig = plt.figure(figsize=(6.4, 6.9))
    ax = fig.add_axes([0.135, 0.045, 0.795, 0.780])
    legend_below(ax, fn(ax, "train", NINE, letter), ncol=1, fs=6.6)
    save_fig(fig, f"FigS1S_TrainOOF_{tag}")
    plt.close(fig)
    print(f"[SAVED] FigS1S_TrainOOF_{tag}")

# ==========================================================================
# 4. Fig 4 C/G — wide PR (stacking + top-3, AP CI)
# ==========================================================================
MODELS4 = ["Stacking"] + TOP3
fig = plt.figure(figsize=(12.6, 5.8))
axA = fig.add_axes([0.058, 0.045, 0.372, 0.795])
axB = fig.add_axes([0.556, 0.045, 0.372, 0.795])
legend_below(axA, pr_draw(axA, "test", MODELS4, "C"), ncol=2, fs=6.6)
legend_below(axB, pr_draw(axB, "eICU", MODELS4, "B"), ncol=2, fs=6.6)
fig.text(0.04, 0.985,
         "Precision-recall — stacking ensemble vs the three selected base "
         "learners", ha="left", va="top",
         fontsize=10.2, fontweight="bold")
save_fig(fig, "Fig4CG_PR_two_cohorts")
plt.close(fig)
print("[SAVED] Fig4CG_PR_two_cohorts")

# ==========================================================================
# 5. Fig 4 D/H — wide DCA with NB CI box
# ==========================================================================
fig = plt.figure(figsize=(12.6, 5.8))
axA = fig.add_axes([0.058, 0.075, 0.372, 0.765])
axB = fig.add_axes([0.556, 0.075, 0.372, 0.765])
hA, _ = dca_draw(axA, "test", MODELS4, "D")
legend_below(axA, hA, ncol=1, fs=6.4)
hB, _ = dca_draw(axB, "eICU", MODELS4, "C")
legend_below(axB, hB, ncol=1, fs=6.4)
fig.text(0.04, 0.985,
         "Decision-curve analysis — stacking ensemble vs the three selected "
         "base learners", ha="left", va="top",
         fontsize=10.2, fontweight="bold")
save_fig(fig, "Fig4DH_DCA_two_cohorts")
plt.close(fig)
print("[SAVED] Fig4DH_DCA_two_cohorts")

# ==========================================================================
# 6. Fig 4 E/I — calibration with CI box
# ==========================================================================
fig = plt.figure(figsize=(12.6, 5.8))
axA = fig.add_axes([0.075, 0.075, 0.355, 0.765])
axB = fig.add_axes([0.575, 0.075, 0.355, 0.765])
hA, _ = calib_draw(axA, "test", "E")
legend_below(axA, hA, ncol=1, fs=6.8)
hB, _ = calib_draw(axB, "eICU", "D")
legend_below(axB, hB, ncol=1, fs=6.8)
fig.text(0.04, 0.985,
         "Calibration of the stacking ensemble — internal test set and eICU "
         "external cohort", ha="left", va="top",
         fontsize=10.2, fontweight="bold")
save_fig(fig, "Fig4EI_calibration_two_cohorts")
plt.close(fig)
print("[SAVED] Fig4EI_calibration_two_cohorts")

# ==========================================================================
# 7. Fig 6 / Fig 7 — ROC | PR | DCA composites with CI
# ==========================================================================
for stem, coh, letters in (("Fig6_ROC_PR_DCA_internal", "test", "BCD"),
                           ("Fig7_ROC_PR_DCA_eICU", "eICU", "ABC")):
    fig = plt.figure(figsize=(12.6, 5.9))
    ax1 = fig.add_axes([0.052, 0.135, 0.205, 0.715])
    ax2 = fig.add_axes([0.335, 0.135, 0.205, 0.715])
    ax3 = fig.add_axes([0.615, 0.135, 0.300, 0.715])
    legend_below(ax1, roc_draw(ax1, coh, MODELS4, letters[0]), ncol=1,
                 fs=6.2)
    legend_below(ax2, pr_draw(ax2, coh, MODELS4, letters[1]), ncol=1,
                 fs=6.2)
    h3, _ = dca_draw(ax3, coh, MODELS4, letters[2])
    legend_below(ax3, h3, ncol=2, fs=6.2)
    fig.text(0.04, 0.985,
             ("Internal hold-out test set" if coh == "test"
              else "External validation (eICU-CRD)")
             + " — ROC, precision-recall and decision-curve analysis "
             "(95% CI in legends)",
             ha="left", va="top", fontsize=10.0, fontweight="bold")
    save_fig(fig, stem)
    plt.close(fig)
    print(f"[SAVED] {stem}")

# ==========================================================================
# 8. Fig 6S / 7S singles
# ==========================================================================
SINGLES = [("Fig6S_ROC_internal", "test", roc_draw, ""),
           ("Fig6S_PR_internal", "test", pr_draw, ""),
           ("Fig7S_ROC_eICU", "eICU", roc_draw, ""),
           ("Fig7S_PR_eICU", "eICU", pr_draw, "")]
for stem, coh, fn, letter in SINGLES:
    fig = plt.figure(figsize=(6.6, 6.9))
    ax = fig.add_axes([0.135, 0.045, 0.795, 0.780])
    legend_below(ax, fn(ax, coh, MODELS4, letter), ncol=1, fs=6.5)
    save_fig(fig, stem)
    plt.close(fig)
    print(f"[SAVED] {stem}")
for stem, coh, letter in (("Fig6S_DCA_internal", "test", ""),
                          ("Fig7S_DCA_eICU", "eICU", "")):
    fig = plt.figure(figsize=(7.4, 6.9))
    ax = fig.add_axes([0.125, 0.045, 0.805, 0.780])
    h, _ = dca_draw(ax, coh, MODELS4, letter)
    legend_below(ax, h, ncol=1, fs=6.5)
    save_fig(fig, stem)
    plt.close(fig)
    print(f"[SAVED] {stem}")

# ==========================================================================
# 9. Supplementary S4 — paired DeLong forest with CI
# ==========================================================================
fig = plt.figure(figsize=(12.6, 5.9))
for k, (coh, letter) in enumerate((("test", "A"), ("eICU", "B"))):
    ax = fig.add_axes([0.088 + k * 0.475, 0.135, 0.330, 0.700])
    sub = DL[DL.cohort == coh].sort_values("delta")
    yy = np.arange(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        col = "#B2182B" if r.p < 0.05 else "#9A9A9A"
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.4, zorder=3)
        ax.plot(r.delta, i, "o", color=col, ms=4.0, zorder=4)
    ax.axvline(0, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([FULL[b] for b in sub["base"]], fontsize=7.2)
    xmax = max(0.068, float(sub["hi"].max()) * 1.9)
    ax.set_xlim(-0.006, xmax)
    for i, (_, r) in enumerate(sub.iterrows()):
        star = " *" if r.p < 0.05 else "  NS"
        pv = f"P = {r.p:.1e}" if r.p >= 1e-4 else "P < 0.001"
        ax.text(r.hi + xmax * 0.02, i,
                f"+{r.delta:.3f} (+{r.lo:.3f} to +{r.hi:.3f})   {pv}{star}",
                va="center", ha="left", fontsize=6.3,
                color="#B2182B" if r.p < 0.05 else "#7A7A7A")
    ax.set_xlabel("\u0394AUC (stacking \u2212 base learner), DeLong 95% CI")
    ax.set_title(f"{letter}   {'Internal hold-out test set' if coh == 'test' else 'eICU-CRD external cohort'}",
                 loc="left", fontsize=9.0, fontweight="bold", pad=5)
fig.text(0.04, 0.985,
         "Paired DeLong tests — stacking ensemble vs every base learner "
         "  (* P < 0.05)",
         ha="left", va="top", fontsize=10.2, fontweight="bold")
save_fig(fig, "FigS4_delong_forest")
plt.close(fig)
print("[SAVED] FigS4_delong_forest")
print("[ALL DONE]")
