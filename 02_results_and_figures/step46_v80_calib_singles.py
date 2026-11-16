# -*- coding: utf-8 -*-
"""step46_v80_calib_singles.py — per-cohort calibration singles for the
manuscript Figure 4 (internal) and Figure 5 (external) composites."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import (  # noqa: E402
    V80_DIR, COLORS, set_style, save_fig,
)

plt = set_style()
from sklearn.linear_model import LogisticRegression  # noqa: E402

D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
Y = {"test": D["y_test"].astype(int), "eICU": D["y_eicu"].astype(int)}
PFX = {"test": "tst__", "eICU": "ext__"}
NTITLE = {
    "test": "Internal hold-out test set (n = 10,316)",
    "eICU": "eICU-CRD external cohort (n = 32,067)"}
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv",
                  encoding="utf-8-sig").set_index("cohort")


def calib_draw(ax, coh, letter):
    y, s = Y[coh], D[PFX[coh] + "Stacking"]
    eps = 1e-6
    lp = np.log(np.clip(s, eps, 1 - eps) / (1 - np.clip(s, eps, 1 - eps)))
    qs = np.quantile(s, np.linspace(0, 1, 11))
    xs, ys_ = [], []
    for j in range(10):
        m_ = (s >= qs[j]) & (s < qs[j + 1] if j < 9 else s <= qs[j + 1])
        if m_.sum() < 30:
            continue
        xs.append(float(np.mean(s[m_])))
        ys_.append(float(np.mean(y[m_])))
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    grid = np.linspace(0.005, 0.995, 60)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(400):
        ix = np.concatenate([rng.choice(pos, len(pos), True),
                             rng.choice(neg, len(neg), True)])
        lrb = LogisticRegression(C=1e6, max_iter=500).fit(
            lp[ix].reshape(-1, 1), y[ix])
        boots.append(lrb.predict_proba(
            np.log(grid / (1 - grid)).reshape(-1, 1))[:, 1])
    boots = np.asarray(boots)
    blo, bhi = np.percentile(boots, [2.5, 97.5], axis=0)
    ax.fill_between(grid, blo, bhi, color="#D64545", alpha=0.18, lw=0,
                    zorder=2)
    ax.plot(grid, np.median(boots, axis=0), color="#B2182B", lw=1.6,
            zorder=3)
    ax.plot(xs, ys_, "o", color="#4A4A4A", ms=3.4, zorder=4)
    ax.plot([0, 1], [0, 1], color="#808080", lw=0.9, ls=(0, (5, 3)), zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted probability of NOAF")
    ax.set_ylabel("Observed proportion of NOAF")
    ax.set_title(f"{letter + '   ' if letter else ''}{NTITLE[coh]}", loc="left", pad=5,
                 fontsize=9.0, fontweight="bold")
    r = CAL.loc[coh]
    ax.text(0.03, 0.97,
            f"Intercept {r['intercept']:+.3f} ({r['i_lo']:+.3f} to "
            f"{r['i_hi']:+.3f})\n"
            f"Slope {r['slope']:.3f} ({r['s_lo']:.3f} to {r['s_hi']:.3f})\n"
            f"Brier {r.brier:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.4, alpha=0.96))
    return ax


set_style()
for coh, stem, letter in (("test", "Fig4E_calibration_internal", ""),
                          ("eICU", "Fig5D_calibration_eICU", "")):
    fig = plt.figure(figsize=(6.6, 6.6))
    ax = fig.add_axes([0.135, 0.045, 0.795, 0.780])
    calib_draw(ax, coh, letter)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color="#808080", lw=0.9, ls=(0, (5, 3)),
               label="Ideal (slope 1, intercept 0)"),
        Line2D([], [], color="#B2182B", lw=1.6,
               label="Stacking ensemble (calibrated)"),
        Line2D([], [], color="#4A4A4A", marker="o", ls="", ms=3.4,
               label="Deciles of predicted risk")],
        loc="upper left", bbox_to_anchor=(0.0, -0.115), frameon=False,
        fontsize=7.0)
    save_fig(fig, stem)
    plt.close(fig)
    print(f"[SAVED] {stem}")
print("[DONE]")
