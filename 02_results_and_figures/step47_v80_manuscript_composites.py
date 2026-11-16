# -*- coding: utf-8 -*-
"""step47_v80_manuscript_composites.py — PIL composites for the manuscript.

v2: panel letters are drawn ON THE CANVAS at a fixed offset with ONE shared
font size, and every panel is pasted left-aligned inside its grid column, so
letters align vertically within a column and horizontally within a row.

  Figure4_stacking_internal : A workflow | B ROC | C PR / D DCA | E calibration
  Figure5_stacking_external : A ROC | B PR / C DCA | D calibration
  Figure6_SHAP              : A beeswarm | B TP / C TN | D FP / E FN | F stability
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR, FIG_DIR  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402

FONT_BOLD = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def load(name):
    p = FIG_DIR / f"{name}.png"
    im = Image.open(p)
    if im.mode in ("RGBA", "P"):
        im = im.convert("RGB")
    return im


def tag(canvas, xy, tag, fs):
    """Draw one bold panel letter on the canvas at absolute position xy."""
    font = ImageFont.truetype(FONT_BOLD, fs)
    d = ImageDraw.Draw(canvas)
    d.text(xy, tag, font=font, fill="#111111",
           stroke_width=max(2, fs // 18), stroke_fill="#FFFFFF")


def build_grid(cells, out_stem, gap=24, margin=30):
    """cells: dict {(row, col): (image, letter)}; columns left-aligned.

    Letter font size is shared across the whole figure; letters are drawn at
    (panel_x + pad, panel_y + pad) -> aligned per column and per row.
    """
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
    pos = {}
    y = margin
    for r in range(n_rows):
        x = margin
        for c in range(n_cols):
            if (r, c) in cells:
                im, letter = cells[(r, c)]
                canvas.paste(im, (x, y))
                tag(canvas, (x + pad, y + pad), letter, fs)
                pos[letter] = (x + pad, y + pad)
            x += col_w[c] + gap
        y += row_h[r] + gap
    out = FIG_DIR / f"{out_stem}.png"
    canvas.save(out, dpi=(400, 400))
    print("[SAVED]", out, f"letter fs={fs}, positions={pos}")


# ---------------- Figure 4: workflow + internal test ------------------------
roc_i = load("Fig6S_ROC_internal")
pr_i = load("Fig6S_PR_internal")
dca_i = load("Fig6S_DCA_internal")
cal_i = load("Fig4E_calibration_internal")

gap, margin = 24, 30
col0_w = max(roc_i.width, dca_i.width)
col1_w = max(pr_i.width, cal_i.width)
grid_w = col0_w + gap + col1_w
wf = load("Figure_Workflow_stacking")
s = grid_w / wf.width
wf = wf.resize((grid_w, int(wf.height * s)), Image.LANCZOS)

cells4 = {
    (0, 0): (wf, "A"),
    (1, 0): (roc_i, "B"),
    (1, 1): (pr_i, "C"),
    (2, 0): (dca_i, "D"),
    (2, 1): (cal_i, "E"),
}
build_grid(cells4, "Figure4_stacking_internal", gap=gap, margin=margin)

# ---------------- Figure 5: external validation -----------------------------
cells5 = {
    (0, 0): (load("Fig7S_ROC_eICU"), "A"),
    (0, 1): (load("Fig7S_PR_eICU"), "B"),
    (1, 0): (load("Fig7S_DCA_eICU"), "C"),
    (1, 1): (load("Fig5D_calibration_eICU"), "D"),
}
build_grid(cells5, "Figure5_stacking_external", gap=gap, margin=margin)

# ---------------- Figure 6: SHAP (needs step44 outputs) ---------------------
need = ["Fig8_SHAP_beeswarm", "Fig8_SHAP_waterfall_TP", "Fig8_SHAP_waterfall_TN",
        "Fig8_SHAP_waterfall_FP", "Fig8_SHAP_waterfall_FN",
        "Fig8_SHAP_stability"]
if all((FIG_DIR / f"{n}.png").exists() for n in need):
    def norm(im, w):
        sc = w / im.width
        return im.resize((w, int(im.height * sc)), Image.LANCZOS)
    WCOL = min(load(n).width for n in need)
    cells6 = {
        (0, 0): (norm(load("Fig8_SHAP_beeswarm"), WCOL), "A"),
        (0, 1): (norm(load("Fig8_SHAP_waterfall_TP"), WCOL), "B"),
        (1, 0): (norm(load("Fig8_SHAP_waterfall_TN"), WCOL), "C"),
        (1, 1): (norm(load("Fig8_SHAP_waterfall_FP"), WCOL), "D"),
        (2, 0): (norm(load("Fig8_SHAP_waterfall_FN"), WCOL), "E"),
        (2, 1): (norm(load("Fig8_SHAP_stability"), WCOL), "F"),
    }
    build_grid(cells6, "Figure6_SHAP", gap=gap, margin=margin)
else:
    print("[SKIP] Figure6_SHAP — waiting for step44 outputs")
print("[DONE]")
