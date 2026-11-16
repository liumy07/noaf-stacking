# -*- coding: utf-8 -*-
"""step65_fix_fig6_alignment.py — make Figure 6F flush with 6E (bottom-aligned).

Problem: Fig8_SHAP_stability was drawn 1x2 (aspect 2.10). In the Figure 6
composite every panel is normalised to the beeswarm column width, so panel F
ended up ~1000 px shorter than the waterfalls and build_grid's top-aligned
pasting left a white block at the bottom-right of the manuscript figure.

Fix: redraw Fig8_SHAP_stability as 2x1 (target aspect 1.20 == waterfall FN
aspect) so row 3 of the composite is naturally flush, then rebuild
Figure6_SHAP.png with the same build_grid logic as step47.

Inputs  : v90_primary/v80_shap_stability.csv, v80_shap_stability_summary.csv,
          v80_shap_ranking.csv           (all produced by step44, unchanged)
Outputs : v90_primary/figures/Fig8_SHAP_stability.(png|pdf|svg)
          v90_primary/figures/Figure6_SHAP.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NOAF_RUN", "v90")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

from noaf_common_v80 import V80_DIR, FIG_DIR, LABELS, set_style, save_fig  # noqa: E402

# ----------------------------------------------------------------------------
# 1. stability panel, redrawn vertically (2 rows x 1 col)
# ----------------------------------------------------------------------------
stab = pd.read_csv(V80_DIR / "v80_shap_stability.csv", index_col=0).T
summary = pd.read_csv(V80_DIR / "v80_shap_stability_summary.csv")
rk = pd.read_csv(V80_DIR / "v80_shap_ranking.csv")

feat_names = list(stab.columns)
base_rank = pd.Series(rk["MeanAbsSHAP"].values, index=rk["Feature"]).rank(
    ascending=False)
N_RUNS = len(stab)
rhos = [spearmanr(base_rank[feat_names].values, stab.iloc[r][feat_names].values)
        .statistic for r in range(N_RUNS)]
print(f"[CHECK] rho mean {np.mean(rhos):.4f} min {np.min(rhos):.4f} "
      f"max {np.max(rhos):.4f}  (must match manuscript 0.977 / 0.969)")

plt = set_style()
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0))   # aspect 1.20

ax = axes[0]
TN_ = 15
sub = summary.head(TN_)
yv = np.arange(TN_)[::-1]
lab_map = dict(zip(rk["Feature"], rk["Label"]))
for y_, (_, r) in zip(yv, sub.iterrows()):
    ax.plot([r["MinRank"], r["MaxRank"]], [y_, y_], color="#BBBBBB",
            lw=2.4, solid_capstyle="round", zorder=2)
    ax.plot(r["MedianRank"], y_, "o", ms=4.0, color="#D62728", zorder=3)
    ax.plot(base_rank[r["Feature"]], y_, "|", ms=7, color="#1F6FB4",
            mew=1.4, zorder=4)
ax.set_yticks(yv)
ax.set_yticklabels([lab_map[f] for f in sub["Feature"]], fontsize=8)
ax.set_xlabel("Feature rank (1 = most important)")
ax.set_title(f"A  Rank stability across {N_RUNS} explain sets",
             loc="left", fontweight="bold", fontsize=9.5)
ax.legend(handles=[Line2D([], [], color="#BBBBBB", lw=2.4,
                          label="Rank range across runs"),
                   Line2D([], [], color="#D62728", marker="o", ls="", ms=4.0,
                          label="Median rank"),
                   Line2D([], [], color="#1F6FB4", marker="|", ls="", ms=7,
                          mew=1.4, label="Main analysis (seed 42)")],
          loc="upper left", bbox_to_anchor=(0.0, -0.18), fontsize=7.2,
          ncol=3, frameon=False)
ax.invert_xaxis()

ax = axes[1]
ax.hist(rhos, bins=10, color="#4E79A7", edgecolor="white", zorder=2)
ax.axvline(np.mean(rhos), ls="--", color="#D62728", lw=1.2, zorder=3)
ax.text(np.mean(rhos), ax.get_ylim()[1] * 0.92,
        f"mean = {np.mean(rhos):.3f}", ha="left", va="top", fontsize=8,
        color="#D62728")
ax.set_xlabel("Spearman \u03c1 vs main analysis")
ax.set_ylabel("Number of runs")
ax.set_title("B  Ranking agreement", loc="left", fontweight="bold",
             fontsize=9.5)
fig.tight_layout()
save_fig(fig, "Fig8_SHAP_stability")
plt.close(fig)

im = Image.open(FIG_DIR / "Fig8_SHAP_stability.png")
print(f"[SAVED] Fig8_SHAP_stability {im.size}  aspect {im.width/im.height:.3f} "
      f"(target 1.200)")

# ----------------------------------------------------------------------------
# 2. rebuild the Figure 6 composite (same logic as step47.build_grid)
# ----------------------------------------------------------------------------
FONT_BOLD = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def load(name):
    p = FIG_DIR / f"{name}.png"
    im = Image.open(p)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    return im


def norm(im, w):
    sc = w / im.width
    return im.resize((w, int(im.height * sc)), Image.LANCZOS)


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
    for (r, c), (im, _) in cells.items():
        col_w[c] = max(col_w[c], im.width)
        row_h[r] = max(row_h[r], im.height)
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
                im, letter = cells[(r, c)]
                canvas.paste(im, (x, y))
                tag(canvas, (x + pad, y + pad), letter, fs)
            x += col_w[c] + gap
        y += row_h[r] + gap
    out = FIG_DIR / f"{out_stem}.png"
    canvas.save(out, dpi=(400, 400))
    print(f"[SAVED] {out} {canvas.size}  rows={row_h}")


need = ["Fig8_SHAP_beeswarm", "Fig8_SHAP_waterfall_TP", "Fig8_SHAP_waterfall_TN",
        "Fig8_SHAP_waterfall_FP", "Fig8_SHAP_waterfall_FN",
        "Fig8_SHAP_stability"]
WCOL = min(load(n).width for n in need)
cells6 = {
    (0, 0): (norm(load("Fig8_SHAP_beeswarm"), WCOL), "A"),
    (0, 1): (norm(load("Fig8_SHAP_waterfall_TP"), WCOL), "B"),
    (1, 0): (norm(load("Fig8_SHAP_waterfall_TN"), WCOL), "C"),
    (1, 1): (norm(load("Fig8_SHAP_waterfall_FP"), WCOL), "D"),
    (2, 0): (norm(load("Fig8_SHAP_waterfall_FN"), WCOL), "E"),
    (2, 1): (norm(load("Fig8_SHAP_stability"), WCOL), "F"),
}
build_grid(cells6, "Figure6_SHAP")
print("[DONE]")
