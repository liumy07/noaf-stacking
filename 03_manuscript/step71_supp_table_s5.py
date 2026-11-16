# -*- coding: utf-8 -*-
"""
step71_supp_table_s5.py — rebuild Supplementary Table S5 (calibration metrics).

The original S5 file is not available on this machine.  This script recomputes
calibration for the stacking ensemble AND its three base learners in both
cohorts with the exact estimator used by step41_v80_ci.py:

    logit(P) = alpha + beta * logit(p_hat)      (joint logistic recalibration)
    - alpha : calibration intercept (calibration-in-the-large)
    - beta  : calibration slope
    - Brier : mean((p_hat - y)^2)
    - E:O   : mean(p_hat) / mean(y)
    CIs     : percentile bootstrap, 1000 resamples stratified by outcome,
              alpha and beta re-estimated jointly in every resample.

Sanity gate: the stacking rows must reproduce v80_ci_calibration.csv to <1e-9;
otherwise the script aborts instead of writing a table that disagrees with the
figures.

Outputs
    v90_primary/tables/v80_calibration_all_models.csv   (traceability)
    Supplementary_Table_S5_calibration.docx            (project folder)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ("00_common", "02_results_and_figures"):
    sys.path.insert(0, str(ROOT / sub))
from noaf_common_v80 import V80_DIR  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402

SEED, NBOOT = 42, 1000
D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
TOP3 = [str(x) for x in D["top3"]]
Y = {"test": D["y_test"].astype(int), "eICU": D["y_eicu"].astype(int)}
PFX = {"test": "tst__", "eICU": "ext__"}
LABEL = {
    "Stacking": "Stacking ensemble",
    "MLP": "Multilayer perceptron",
    "XGBoost": "XGBoost",
    "RF": "Random forest",
}


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def boot_idx(y, rng):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos), True),
                           rng.choice(neg, len(neg), True)])


def fit_cal(yy, lpp):
    lr = LogisticRegression(C=1e6, max_iter=500).fit(lpp.reshape(-1, 1), yy)
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


rows = []
for coh in ("test", "eICU"):
    y = Y[coh]
    for m in ["Stacking"] + TOP3:
        s = D[PFX[coh] + m].astype(float)
        lp = logit(s)
        a0, b0 = fit_cal(y, lp)
        rng = np.random.default_rng(SEED + 2)
        out = np.empty((NBOOT, 2))
        for b in range(NBOOT):
            ix = boot_idx(y, rng)
            out[b] = fit_cal(y[ix], lp[ix])
        a_lo, a_hi = np.percentile(out[:, 0], [2.5, 97.5])
        b_lo, b_hi = np.percentile(out[:, 1], [2.5, 97.5])
        rows.append(dict(
            cohort=coh, model=m, n=len(y),
            intercept=a0, i_lo=a_lo, i_hi=a_hi,
            slope=b0, s_lo=b_lo, s_hi=b_hi,
            brier=float(np.mean((s - y) ** 2)),
            E_O=float(np.mean(s) / np.mean(y))))
        print(f"{coh:5s} {m:9s} int {a0:+.4f} ({a_lo:+.4f}, {a_hi:+.4f})  "
              f"slope {b0:.4f} ({b_lo:.4f}, {b_hi:.4f})  "
              f"Brier {rows[-1]['brier']:.4f}  E:O {rows[-1]['E_O']:.3f}",
              flush=True)

tab = pd.DataFrame(rows)
outdir = V80_DIR / "tables"
outdir.mkdir(parents=True, exist_ok=True)
tab.to_csv(outdir / "v80_calibration_all_models.csv", index=False,
           encoding="utf-8-sig")

# ---------------------------------------------------------- sanity gate
ref = pd.read_csv(V80_DIR / "v80_ci_calibration.csv",
                  encoding="utf-8-sig").set_index("cohort")
for coh in ("test", "eICU"):
    got = tab[(tab.cohort == coh) & (tab.model == "Stacking")].iloc[0]
    for col in ("intercept", "slope", "brier", "E_O"):
        d = abs(float(got[col]) - float(ref.loc[coh, col]))
        assert d < 1e-9, f"{coh}.{col} mismatch {d:.2e}"
print("\n[OK] stacking rows reproduce v80_ci_calibration.csv exactly")

# ------------------------------------------------------------- build docx
import docx  # noqa: E402
from docx.enum.section import WD_ORIENT  # noqa: F401,E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.oxml import OxmlElement  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.shared import Inches, Pt  # noqa: E402

doc = docx.Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for attr in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, attr, Inches(1))

normal = doc.styles["Normal"]
normal.font.name = "Times New Roman"
normal.font.size = Pt(12)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def add_par(text="", size=12, bold=False, italic=False, space_before=0,
            space_after=6, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.0
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p


add_par("Supplementary Table S5. Calibration of the stacking ensemble and of its "
        "three base learners in the internal hold-out test set and in the "
        "external eICU-CRD cohort.", size=10, bold=True, space_after=8)


def fmt_ci(est, lo, hi, sign=False):
    f = (lambda v: f"{v:+.3f}") if sign else (lambda v: f"{v:.3f}")
    return f"{f(est)} ({f(lo)}, {f(hi)})"


def set_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    for i, seg in enumerate(text.split("\n")):
        if i:
            p.add_run().add_break()
        r = p.add_run(seg)
        r.font.name = "Times New Roman"
        r.font.size = Pt(9)
        r.bold = bold


HEAD = ["Model", "Cohort", "Calibration\nintercept (95% CI)",
        "Calibration\nslope (95% CI)", "Brier\nscore", "E:O\nratio"]
WIDTHS = [1850, 1500, 2400, 2100, 700, 780]

tbl = doc.add_table(rows=1, cols=len(HEAD))
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl.autofit = False

# three-line table: outer top/bottom 12 (1.5 pt), header bottom 6 (0.75 pt)
tblPr = tbl._tbl.tblPr
borders = OxmlElement("w:tblBorders")
for tag, sz in (("top", 12), ("bottom", 12)):
    el = OxmlElement(f"w:{tag}")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:space"), "0")
    el.set(qn("w:color"), "auto")
    borders.append(el)
tblPr.append(borders)
layout = OxmlElement("w:tblLayout")
layout.set(qn("w:type"), "fixed")
tblPr.append(layout)

hdr = tbl.rows[0]
for i, h in enumerate(HEAD):
    set_cell(hdr.cells[i], h, bold=True)
    tcPr = hdr.cells[i]._tc.get_or_add_tcPr()
    b = OxmlElement("w:tcBorders")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"), "single")
    bot.set(qn("w:sz"), "6")
    bot.set(qn("w:space"), "0")
    bot.set(qn("w:color"), "auto")
    b.append(bot)
    tcPr.append(b)

COHORT_TXT = {"test": "Internal test set\n(n = {:,})",
              "eICU": "eICU-CRD\nexternal cohort\n(n = {:,})"}
for coh in ("test", "eICU"):
    for m in ["Stacking"] + TOP3:
        r = tab[(tab.cohort == coh) & (tab.model == m)].iloc[0]
        cells = tbl.add_row().cells
        set_cell(cells[0], LABEL[m],
                 align=WD_ALIGN_PARAGRAPH.LEFT)
        set_cell(cells[1], COHORT_TXT[coh].format(int(r.n)))
        set_cell(cells[2], fmt_ci(r.intercept, r.i_lo, r.i_hi, sign=True))
        set_cell(cells[3], fmt_ci(r.slope, r.s_lo, r.s_hi))
        set_cell(cells[4], f"{r.brier:.3f}")
        set_cell(cells[5], f"{r.E_O:.2f}")

for row in tbl.rows:
    for i, c in enumerate(row.cells):
        c.width = Pt(WIDTHS[i] / 20)
for i, w in enumerate(WIDTHS):
    tbl.columns[i].width = Pt(w / 20)

add_par("", size=6, space_after=2)
add_par(
    "Calibration was quantified by fitting the logistic recalibration model "
    "logit(P) = \u03b1 + \u03b2\u00b7logit(p), in which \u03b1 is the "
    "calibration intercept (calibration-in-the-large; 0 for a perfectly "
    "calibrated model) and \u03b2 the calibration slope (1 for a perfectly "
    "calibrated model); p denotes the predicted probability from the model. "
    "The Brier score is the mean squared difference between "
    "the predicted probability and the observed outcome. The expected:observed "
    "(E:O) ratio is the mean predicted probability divided by the observed "
    "event rate. Confidence intervals are percentile intervals from 1,000 "
    "bootstrap resamples stratified by outcome, with \u03b1 and \u03b2 "
    "re-estimated jointly in each resample. Stacking-ensemble rows correspond "
    "to the calibration panels of Figures 4E and 5D. CI, confidence interval.",
    size=8, space_before=4, space_after=0)

OUT = Path(r"D:/BJND-4002-11项目/BJND-4002-11/Supplementary_Table_S5_calibration.docx")
doc.save(str(OUT))
print(f"[OK] wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB)")
