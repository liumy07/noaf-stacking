# -*- coding: utf-8 -*-
"""step66_fix_fig5d_alignment.py — make Figure 5D flush with 5C (bottom-aligned).

Problem: in Figure5_stacking_external, row 2 contains C = Fig7S_DCA_eICU
(2678 x 2727 px) and D = Fig5D_calibration_eICU (2290 x 2583 px).  build_grid
pastes top-aligned within the row, so D ended 144 px short of C and a white
block was left at the bottom-right of the manuscript figure.

Fix: redraw Fig5D_calibration_eICU at figsize (6.6, 6.96) instead of
(6.6, 6.6) so the tight-cropped panel is ~2290 x 2727 (same height as C),
then rebuild Figure5_stacking_external.png with the same build_grid logic as
step47.  Statistics shown (Intercept -1.734, Slope 0.722, Brier 0.1068) come
from the unchanged v80_ci_calibration.csv / v80_scores.npz.

Inputs  : v90_primary/v80_scores.npz, v80_ci_calibration.csv   (step41/46 outputs)
Outputs : v90_primary/figures/Fig5D_calibration_eICU.(png|pdf|svg)
          v90_primary/figures/Figure5_stacking_external.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NOAF_RUN", "v90")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image

from noaf_common_v80 import V80_DIR, FIG_DIR, set_style, save_fig  # noqa: E402

plt = set_style()
from sklearn.linear_model import LogisticRegression  # noqa: E402

# ----------------------------------------------------------------------------
# 1. redraw the eICU calibration panel, taller (target height 2727 px)
# ----------------------------------------------------------------------------
D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
Y = {"eICU": D["y_eicu"].astype(int)}
PFX = {"eICU": "ext__"}
NTITLE = {"eICU": "eICU-CRD external cohort (n = 32,067)"}
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv",
                  encoding="utf-8-sig").set_index("cohort")


def calib_draw(ax, coh):
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
    ax.set_title(NTITLE[coh], loc="left", pad=5,
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


TARGET_H = 2727          # == Fig7S_DCA_eICU height (row-mate of panel D)
TARGET_W = 2290          # == Fig7S_ROC/PR_eICU width (keeps column widths)
H_IN = 6.96              # 6.6 -> 6.96 gives +144 px at 400 dpi

stem = "Fig5D_calibration_eICU"
fig = plt.figure(figsize=(6.6, H_IN))
ax = fig.add_axes([0.135, 0.045, 0.795, 0.780])
calib_draw(ax, "eICU")
from matplotlib.lines import Line2D  # noqa: E402
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

im = Image.open(FIG_DIR / f"{stem}.png")
print(f"[SAVED] {stem} {im.size}  aspect {im.width/im.height:.3f} "
      f"(target {TARGET_W} x {TARGET_H}, aspect {TARGET_W/TARGET_H:.3f})")

# normalise to the exact cell size (content top-aligned, white fill)
if im.size != (TARGET_W, TARGET_H):
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), "#FFFFFF")
    w = min(im.width, TARGET_W)
    h = min(im.height, TARGET_H)
    canvas.paste(im.convert("RGB").crop((0, 0, w, h)), (0, 0))
    canvas.save(FIG_DIR / f"{stem}.png", dpi=(400, 400))
    print(f"[PADDED] {stem} -> {TARGET_W} x {TARGET_H}")
    im = canvas

# ----------------------------------------------------------------------------
# 2. rebuild the Figure 5 composite (same logic as step47.build_grid)
# ----------------------------------------------------------------------------
import matplotlib.font_manager as fm  # noqa: E402
from PIL import ImageDraw, ImageFont  # noqa: E402

FONT_BOLD = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def load(name):
    p = FIG_DIR / f"{name}.png"
    img = Image.open(p)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img


def tag(canvas, xy, letter, fs):
    font = ImageFont.truetype(FONT_BOLD, fs)
    d = ImageDraw.Draw(canvas)
    d.text(xy, letter, font=font, fill="#111111",
           stroke_width=max(2, fs // 18), stroke_fill="#FFFFFF")


def build_grid(cells, out_stem, gap=24, margin=30):
    n_rows = max(r for r, _ in cells) + 1
    n_cols = max(c for _, c in cells) + 1
    col_w = [0] * n_cols
    row_h = [0] * n_rows
    for (r, c), (img, _) in cells.items():
        col_w[c] = max(col_w[c], img.width)
        row_h[r] = max(row_h[r], img.height)
    W = margin * 2 + sum(col_w) + gap * (n_cols - 1)
    H = margin * 2 + sum(row_h) + gap * (n_rows - 1)
    canvas = Image.new("RGB", (W, H), "#FFFFFF")
    fs = max(22, int(min(min(row_h), min(col_w)) * 0.028))
    pad = fs // 3
    y = margin
    for r in range(n_rows):
        x = margin
        for c in range(n_cols):
            if (r, c) in cells:
                img, letter = cells[(r, c)]
                canvas.paste(img, (x, y))
                tag(canvas, (x + pad, y + pad), letter, fs)
            x += col_w[c] + gap
        y += row_h[r] + gap
    out = FIG_DIR / f"{out_stem}.png"
    canvas.save(out, dpi=(400, 400))
    print(f"[SAVED] {out} {canvas.size}  rows={row_h}")


gap, margin = 24, 30
cells5 = {
    (0, 0): (load("Fig7S_ROC_eICU"), "A"),
    (0, 1): (load("Fig7S_PR_eICU"), "B"),
    (1, 0): (load("Fig7S_DCA_eICU"), "C"),
    (1, 1): (load("Fig5D_calibration_eICU"), "D"),
}
build_grid(cells5, "Figure5_stacking_external", gap=gap, margin=margin)
print("[DONE]")
