# -*- coding: utf-8 -*-
"""
step57_sync_v90_deck_word.py — bring the deck and the manuscript fully in line
with the current (v80 primary + v90 sensitivity) results:

PPT  (新结果figure设计_修订版.pptx)
  1. NEW slide inserted right after the "Supplementary Table S4 (visualised)"
     forest-plot slide: the actual Supplementary Table S4 as a NATIVE,
     editable two-panel table (internal test | eICU-CRD), 18 paired DeLong
     comparisons, same table style as the Table 2/3 slides.
  2. NEW final slide: v90 sensitivity analysis (imputer refitted inside every
     CV fold) — Suppl. Table S10 headline numbers for the stacking ensemble.

WORD (NOAF_stacking_manuscript_EndNote_revised_v90.docx)
  3. Supplementary Table S4 appended at the end of the document (three-line
     style matching Table 2, 8 pt Arial-free Times body).

All figures were verified identical to the current sources by MD5 — no image
replacement is needed; picture positions are untouched.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Emu, Pt, RGBColor
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor as PRGBColor
from pptx.util import Emu as PEmu, Pt as PPt
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN

REV = Path(r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/rev_noaf")
PPT_PATH = Path(r"<LOCAL_ROOT>/Desktop/A prediction model for new-onset "
                r"atrial fibrillation in critically ill patients based on stacking "
                r"ensemble strategy/新结果figure设计_修订版.pptx")
DOCX_PATH = Path(r"D:/BJND-4002-11项目/BJND-4002-11/"
                 r"NOAF_stacking_manuscript_EndNote_revised_v90.docx")

TBL_STYLE = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"
BLUE = PRGBColor(0x1E, 0x4F, 0xA8)
GREY = PRGBColor(0x55, 0x55, 0x55)
DARK = PRGBColor(0x33, 0x33, 0x33)

DISP = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost", "RF": "Random forest",
        "LightGBM": "LightGBM", "Logistic": "Logistic regression",
        "SVM": "Support vector machine", "Ridge": "Ridge regression",
        "DT": "Decision tree", "KNN": "K-nearest neighbours"}

# =====================================================================
# load data
# =====================================================================
dl = pd.read_csv(REV / "v80_foldwise" / "v80_delong_paired.csv")
sens = pd.read_csv(REV / "v90_fold_impute" / "v90_sensitivity_AUC.csv")
sdel = pd.read_csv(REV / "v90_fold_impute" / "v90_sensitivity_delong.csv")


def fmt_row(r):
    p = "P < 0.001" if r.p < 1e-3 else f"P = {r.p:.3f}"
    return (DISP[r.base], f"{r.delta:+.4f} ({r.lo:+.4f} to {r.hi:+.4f})", p)


rows_test = [fmt_row(r) for _, r in dl[dl.cohort == "test"].iterrows()]
rows_eicu = [fmt_row(r) for _, r in dl[dl.cohort == "eICU"].iterrows()]
assert len(rows_test) == 9 and len(rows_eicu) == 9

# =====================================================================
# PPT helpers
# =====================================================================
prs = Presentation(str(PPT_PATH))
blank_layout = prs.slides[8].slide_layout


def add_title(slide, text):
    box = slide.shapes.add_textbox(PEmu(411480), PEmu(46000), PEmu(11338560),
                                   PEmu(320000))
    tf = box.text_frame
    tf.word_wrap = False
    para = tf.paragraphs[0]
    run = para.add_run()
    run.text = text
    f = run.font
    f.name = "Arial"
    f.size = PPt(13)
    f.bold = True
    f.color.rgb = BLUE
    return box


def add_note(slide, text, top=6455664, size=10):
    box = slide.shapes.add_textbox(PEmu(411480), PEmu(top), PEmu(11338560),
                                   PEmu(360000))
    tf = box.text_frame
    tf.word_wrap = True
    run = tf.paragraphs[0].add_run()
    run.text = text
    f = run.font
    f.name = "Arial"
    f.size = PPt(size)
    f.color.rgb = GREY
    return box


def style_cell(cell, text, bold=False, size=10, color=DARK, align="left"):
    cell.text = text
    para = cell.text_frame.paragraphs[0]
    run = para.runs[0]
    run.font.name = "Arial"
    run.font.size = PPt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    para.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER}[align]
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = PEmu(72000)
    cell.margin_right = PEmu(72000)
    cell.margin_top = PEmu(18000)
    cell.margin_bottom = PEmu(18000)


def add_table(slide, rows, cols, left, top, width, col_w, header):
    gf = slide.shapes.add_table(rows, cols, PEmu(left), PEmu(top), PEmu(width),
                                PEmu(320000 * rows))
    tbl = gf.table
    # apply the same table style as the Table 2/3 slides
    tblPr = tbl._tbl.find(qn("a:tblPr"))
    tblPr.set("firstRow", "1")
    tblPr.set("bandRow", "1")
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = TBL_STYLE
    for j, w in enumerate(col_w):
        tbl.columns[j].width = PEmu(w)
    for j, h in enumerate(header):
        style_cell(tbl.cell(0, j), h, bold=True, size=10, color=PRGBColor(0xFF, 0xFF, 0xFF),
                   align="left" if j == 0 else "left")
    return tbl


# ------------------------------------------------------------------
# slide A — Supplementary Table S4 (native, editable)
# ------------------------------------------------------------------
s4_slide = prs.slides.add_slide(blank_layout)
add_title(s4_slide, "Supplementary Table S4  ·  Paired DeLong comparison — "
                    "stacking ensemble vs. each base learner")
add_note(s4_slide, "Delta AUC = AUC(stacking) − AUC(base learner); positive values "
                   "favour the stacking ensemble. All P values are from the DeLong test "
                   "for correlated ROC curves; CIs are 95%.", top=400000, size=11)

sub_top = 830000
for label, data, left in [("MIMIC-IV internal test set (n = 10,316)", rows_test, 384048),
                          ("eICU-CRD external cohort (n = 32,067)", rows_eicu, 6266032)]:
    lab = s4_slide.shapes.add_textbox(PEmu(left), PEmu(sub_top), PEmu(5541920),
                                      PEmu(260000))
    r = lab.text_frame.paragraphs[0].add_run()
    r.text = label
    r.font.name = "Arial"
    r.font.size = PPt(11)
    r.font.bold = True
    r.font.color.rgb = DARK
    tbl = add_table(s4_slide, 10, 3, left, sub_top + 280000, 5541920,
                    [1740000, 2801920, 1000000],
                    ["Base learner", "ΔAUC (95% CI)", "P value"])
    for i, row in enumerate(data, 1):
        for j, val in enumerate(row):
            style_cell(tbl.cell(i, j), val, bold=False, size=10)

add_note(s4_slide, "The eICU-CRD comparison with XGBoost has ΔAUC +0.0013 with a 95% CI "
                   "of −0.0016 to +0.0043 (P = 0.373), i.e. the confidence interval "
                   "crosses zero: the stacking ensemble and XGBoost are statistically "
                   "indistinguishable in the external cohort.")

# ------------------------------------------------------------------
# slide B — v90 sensitivity analysis (appended at the end)
# ------------------------------------------------------------------
sA = prs.slides.add_slide(blank_layout)
add_title(sA, "Sensitivity analysis  ·  Imputer refitted inside every cross-validation "
              "fold (Supplementary Table S10)")
add_note(sA, "Whole pipeline repeated with the missForest-type imputer refitted on each "
             "fold's training rows before imputing the held-out fold (screening, nine "
             "learners, meta-learner and model ranking all re-run). Stacking ensemble:",
         top=400000, size=11)

hdr = ["Evaluation set", "AUC, primary analysis", "AUC, imputer refitted in folds",
       "ΔAUC (95% CI)", "P"]
cohort_map = [("train OOF", "Training set, out-of-fold (n = 24,068)"),
              ("internal test", "MIMIC-IV internal test (n = 10,316)"),
              ("eICU external", "eICU-CRD external (n = 32,067)")]
tbl = add_table(sA, 4, 5, 384048, 1000000, 11423904,
                [2900000, 2100000, 2500000, 2623904, 1300000], hdr)
tbl.rows[0].height = PEmu(560000)
for i, (key, label) in enumerate(cohort_map, 1):
    a = sens[(sens.Model == "Stacking") & (sens.cohort == key)].iloc[0]
    d = sdel[sdel.cohort == key].iloc[0]
    style_cell(tbl.cell(i, 0), label, bold=False)
    style_cell(tbl.cell(i, 1), f"{a.AUC_v80:.4f}")
    style_cell(tbl.cell(i, 2), f"{a.AUC_v90:.4f}")
    style_cell(tbl.cell(i, 3), f"{d.delta:+.4f} ({d.lo:+.4f} to {d.hi:+.4f})")
    style_cell(tbl.cell(i, 4), "P < 0.001" if d.p < 1e-3 else f"P = {d.p:.3f}")

add_note(sA, "No single model changed by more than 0.001 in any evaluation set (all nine "
             "models: Supplementary Table S10). The published pipeline is retained as the "
             "primary analysis because the imputer is an outcome-blind, unsupervised "
             "transform fitted on training rows only and never sees the evaluation "
             "cohorts; the frozen predictor applied to the two evaluation cohorts is "
             "therefore unchanged in construction.")

# move slide A (currently last-but-one position is fine? no — it is second to last)
# reorder: s4_slide was appended at the end, then sA appended after it.
# Move s4_slide to position right after slide 14 (index 13) -> new index 14.
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
s4_id = ids[-2]          # s4_slide (appended first)
sA_id = ids[-1]          # sensitivity slide (stays at end)
sldIdLst.remove(s4_id)
sldIdLst.insert(14, s4_id)   # after original slide 14 (0-based index 14)

prs.save(str(PPT_PATH))
print("[PPT SAVED]", PPT_PATH.name, "| slides:", len(prs.slides._sldIdLst))

# =====================================================================
# WORD — append Supplementary Table S4
# =====================================================================
doc = Document(str(DOCX_PATH))

# page break before the supplementary section
doc.add_page_break()

h = doc.add_paragraph()
hr = h.add_run("Supplementary Table S4. DeLong test for paired comparison of the area "
               "under the receiver operating characteristic curve: stacking ensemble "
               "vs. each base learner.")
hr.bold = True
hr.font.size = Pt(10)

n_rows = 1 + len(dl)
t = doc.add_table(rows=n_rows, cols=4)
t.autofit = False
widths = [Emu(1930400), Emu(1200000), Emu(1556000), Emu(800000)]

# three-line borders identical to Table 2
tbl = t._tbl
tblPr = tbl.tblPr
borders = etree.SubElement(tblPr, qn("w:tblBorders"))
for edge, sz in (("top", "12"), ("bottom", "12")):
    e = etree.SubElement(borders, qn(f"w:{edge}"))
    e.set(qn("w:val"), "single")
    e.set(qn("w:sz"), sz)
    e.set(qn("w:space"), "0")
    e.set(qn("w:color"), "auto")
# thin rule under the header row
for c in t.rows[0].cells:
    tcPr = c._tc.get_or_add_tcPr()
    tcB = etree.SubElement(tcPr, qn("w:tcBorders"))
    b = etree.SubElement(tcB, qn("w:bottom"))
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), "6")
    b.set(qn("w:space"), "0")
    b.set(qn("w:color"), "auto")

hdr = ["Comparison", "Delta AUC", "95% CI", "P value"]
cohorts = {"test": "MIMIC-IV internal test", "eICU": "eICU-CRD external"}


def fill(cell, text, bold=False, size=8):
    cell.text = ""
    p = cell.paragraphs[0]
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "Times New Roman"


for j, htxt in enumerate(hdr):
    fill(t.rows[0].cells[j], htxt, bold=True)
    t.rows[0].cells[j].width = widths[j]

for i, (_, r) in enumerate(dl.iterrows(), 1):
    pv = "P < 0.001" if r.p < 1e-3 else f"P = {r.p:.3f}"
    vals = [f"Stacking vs. {DISP[r.base]}, {cohorts[r.cohort]}",
            f"{r.delta:+.4f}", f"{r.lo:+.4f} to {r.hi:+.4f}", pv]
    for j, v in enumerate(vals):
        fill(t.rows[i].cells[j], v)
        t.rows[i].cells[j].width = widths[j]

note = doc.add_paragraph()
nr = note.add_run("Delta AUC = AUC(stacking) − AUC(base learner); positive values favour "
                  "the stacking ensemble. All P values are from the DeLong test for "
                  "correlated receiver operating characteristic curves. The eICU-CRD "
                  "comparison with XGBoost has Delta AUC +0.0013 with a 95% CI of "
                  "−0.0016 to +0.0043 (P = 0.373), i.e. the confidence interval crosses "
                  "zero.")
nr.font.size = Pt(8)

doc.save(str(DOCX_PATH))
print("[WORD SAVED]", DOCX_PATH.name)
