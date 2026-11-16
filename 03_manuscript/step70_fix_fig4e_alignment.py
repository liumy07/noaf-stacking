# -*- coding: utf-8 -*-
"""
step70_fix_fig4e_alignment.py — make Figure 4E flush with 4D (bottom-aligned)
and rebuild the whole Figure 4 composite on the uniform panel grid.

Problems fixed
--------------
1. Fig4E_calibration_internal was 2290 x 2583 px while its row-mate
   Fig6S_DCA_internal is 2628 x 2727 -> E ended 144 px short (same defect as
   Figure 5D, fixed in step66).
2. The composite in circulation (7316 x 8822) was built from an older,
   wider generation of panels: column 0 was ~5030 px wide while column 1
   stayed 2290 px, leaving a ~2400 px white channel between the B/D column
   and the C/E column, and shrinking E to ~120 pt on the manuscript page.

Fix
---
* redraw Fig4E_calibration_internal at figsize (6.6, 6.96) -> ~2708 px,
  white-padded to the uniform 2290 x 2727 cell (statistics unchanged:
  intercept +0.115 (+0.068, +0.164), slope 1.142 (1.097, 1.191),
  Brier 0.1398 from v80_ci_calibration.csv);
* rebuild Figure4_stacking_internal.png with the step47 grid logic from the
  CURRENT singles: workflow A capped at 3700 px and centred, all four curve
  panels 2727 px tall, D/E flush at the bottom;
* swap the image inside the manuscript (paragraph 65), size it to fit the
  648 pt text column with its caption on one page;
* align the manuscript slope-CI upper bound with the raw value
  (1.19052 -> 1.191), which the figure legend reports.

Outputs : v90_primary/figures/Fig4E_calibration_internal.(png|pdf|svg)
          v90_primary/figures/Figure4_stacking_internal.png
          manuscript .docx (main + _v90 sync copy)
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("NOAF_RUN", "v90")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "00_common"))

import numpy as np
import pandas as pd
from PIL import Image

from noaf_common_v80 import V80_DIR, FIG_DIR, set_style, save_fig  # noqa: E402

plt = set_style()
from sklearn.linear_model import LogisticRegression  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------------------------------------------------------------------------
# 1. redraw the internal calibration panel, taller
# ---------------------------------------------------------------------------
D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv",
                  encoding="utf-8-sig").set_index("cohort")
NTITLE = {"test": "Internal hold-out test set (n = 10,316)"}


def calib_draw(ax, coh):
    y = D["y_test"].astype(int)
    s = D["tst__Stacking"]
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


TARGET_W, TARGET_H = 2290, 2727   # uniform cell of every curve panel
stem = "Fig4E_calibration_internal"
fig = plt.figure(figsize=(6.6, 6.96))
ax = fig.add_axes([0.135, 0.045, 0.795, 0.780])
calib_draw(ax, "test")
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
print(f"[SAVED] {stem} {im.size}")
if im.size != (TARGET_W, TARGET_H):
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), "#FFFFFF")
    canvas.paste(im.convert("RGB").crop(
        (0, 0, min(im.width, TARGET_W), min(im.height, TARGET_H))), (0, 0))
    canvas.save(FIG_DIR / f"{stem}.png", dpi=(400, 400))
    print(f"[PADDED] {stem} -> {TARGET_W} x {TARGET_H}")

# ---------------------------------------------------------------------------
# 2. rebuild the Figure 4 composite on the uniform grid
# ---------------------------------------------------------------------------
import matplotlib.font_manager as fm  # noqa: E402
from PIL import ImageDraw, ImageFont  # noqa: E402

FONT_BOLD = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))


def load(name):
    img = Image.open(FIG_DIR / f"{name}.png")
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    return img


def tag(canvas, xy, letter, fs):
    font = ImageFont.truetype(FONT_BOLD, fs)
    d = ImageDraw.Draw(canvas)
    d.text(xy, letter, font=font, fill="#111111",
           stroke_width=max(2, fs // 18), stroke_fill="#FFFFFF")


roc_i = load("Fig6S_ROC_internal")          # 2290 x 2727
pr_i = load("Fig6S_PR_internal")            # 2290 x 2727
dca_i = load("Fig6S_DCA_internal")          # 2628 x 2727
cal_i = load("Fig4E_calibration_internal")  # 2290 x 2727

gap, margin = 24, 30
col0_w = max(roc_i.width, dca_i.width)
col1_w = max(pr_i.width, cal_i.width)
grid_w = col0_w + gap + col1_w
H = margin * 2 + 2727 * 2 + gap

wf = load("Figure_Workflow_stacking")       # 3762 x 2482
wf_w = 3700                                  # cap: keeps the page fit
wf = wf.resize((wf_w, int(wf.height * wf_w / wf.width)), Image.LANCZOS)
wf_h = wf.height
H = margin * 2 + wf_h + gap + 2727 + gap + 2727

canvas = Image.new("RGB", (margin * 2 + grid_w, H), "#FFFFFF")
fs = max(22, int(min(2727, col0_w) * 0.028))
pad = fs // 3

# row 1: workflow, centred
x_wf = margin + (grid_w - wf_w) // 2
canvas.paste(wf, (x_wf, margin))
tag(canvas, (margin + pad, margin + pad), "A", fs)
# row 2: B | C
y2 = margin + wf_h + gap
canvas.paste(roc_i, (margin, y2))
tag(canvas, (margin + pad, y2 + pad), "B", fs)
canvas.paste(pr_i, (margin + col0_w + gap, y2))
tag(canvas, (margin + col0_w + gap + pad, y2 + pad), "C", fs)
# row 3: D | E  (both 2727 tall -> flush bottom)
y3 = y2 + 2727 + gap
canvas.paste(dca_i, (margin, y3))
tag(canvas, (margin + pad, y3 + pad), "D", fs)
canvas.paste(cal_i, (margin + col0_w + gap, y3))
tag(canvas, (margin + col0_w + gap + pad, y3 + pad), "E", fs)

out = FIG_DIR / "Figure4_stacking_internal.png"
canvas.save(out, dpi=(400, 400))
print(f"[SAVED] {out} {canvas.size}  aspect {canvas.width/canvas.height:.4f}")

# ---------------------------------------------------------------------------
# 3. swap the image in the manuscript + fix the slope-CI rounding
# ---------------------------------------------------------------------------
import docx  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402

MAIN = Path(r"D:/BJND-4002-11项目/BJND-4002-11/"
            r"NOAF_stacking_manuscript_EndNote_revised.docx")
SYNC = Path(r"D:/BJND-4002-11项目/BJND-4002-11/"
            r"NOAF_stacking_manuscript_EndNote_revised_v90.docx")
TMP = Path(r"D:/BJND-4002-11项目/BJND-4002-11/_tmp_main_fixed.docx")

WIDTH_PT = 350.0                                   # fits 648 pt column + caption
cx = int(WIDTH_PT / 72 * 914400)
cy = int(WIDTH_PT * canvas.height / canvas.width / 72 * 914400)
print(f"[WORD extent] {WIDTH_PT} x {cy/914400*72:.1f} pt")

d = docx.Document(str(MAIN))
fld0 = d.element.xml.count('w:fldCharType="begin"')

# 3a. image paragraph 65
run = d.paragraphs[65].runs[0]
blip = run._r.findall(".//" + qn("a:blip"))[0]
old_rid = blip.get(qn("r:embed"))
new_rid, _ = d.part.get_or_add_image(str(out))
blip.set(qn("r:embed"), new_rid)
for el in d.paragraphs[65]._p.findall(".//" + qn("wp:extent")):
    el.set("cx", str(cx)); el.set("cy", str(cy))
for el in d.paragraphs[65]._p.findall(".//" + qn("a:ext")):
    el.set("cx", str(cx)); el.set("cy", str(cy))
still = any(b.get(qn("r:embed")) == old_rid
            for b in d.element.body.findall(".//" + qn("a:blip")))
if not still and old_rid in d.part.rels:
    d.part.drop_rel(old_rid)
print(f"[IMG] {old_rid} -> {new_rid}")

# 3b. slope CI upper bound 1.190 -> 1.191 (raw 1.1905154879)
p63 = d.paragraphs[63].runs[0]
assert "1.097-1.190" in p63.text
p63.text = p63.text.replace("1.097-1.190", "1.097-1.191")
print("[P63] slope CI -> 1.097-1.191")

fld1 = d.element.xml.count('w:fldCharType="begin"')
assert fld0 == fld1 == 43, (fld0, fld1)
d.save(str(TMP))
print("[SAVED]", TMP)

for target in (MAIN, SYNC):
    for attempt in range(8):
        try:
            shutil.copyfile(TMP, target)
            print("[OK]", target.name)
            break
        except PermissionError:
            time.sleep(3)
    else:
        raise RuntimeError(f"locked: {target}")
