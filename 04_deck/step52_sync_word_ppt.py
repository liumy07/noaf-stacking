# -*- coding: utf-8 -*-
"""step52_sync_word_ppt.py — embed aligned-letter v80 figures in the Word
manuscript and unify PPT caption numbering with the Word scheme (Figure 1-6).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu
from pptx import Presentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import FIG_DIR  # noqa: E402

DOCX = Path(r"D:\BJND-4002-11项目\BJND-4002-11\NOAF_stacking_manuscript_EndNote_revised.docx")
DECK = Path(r"C:\Users\localuser\Desktop\A prediction model for new-onset "
            r"atrial fibrillation in critically ill patients based on stacking "
            r"ensemble strategy\新结果figure设计_修订版.pptx")
OUT = Path(__file__).resolve().parent

# =============================================================== 1. WORD ====
bak = OUT / "_backup_manuscript_pre_figsync.docx"
if not bak.exists():
    shutil.copy2(DOCX, bak)
    print("[BACKUP]", bak.name)

d = Document(str(DOCX))
P = d.paragraphs


def fig_width(p):
    ext = p._element.findall(".//" + qn("wp:extent"))
    return int(ext[0].get("cx")) if ext else Emu(int(15.5 * 360000))


def replace_picture(p, img):
    w = fig_width(p)
    for r in list(p.runs):
        if r._element.findall(".//" + qn("w:drawing")):
            r._element.getparent().remove(r._element)
    run = p.add_run()
    run.add_picture(str(img), width=Emu(w))


for pno, fname in [(43, "Figure_Flowchart_study.png"),
                   (65, "Figure4_stacking_internal.png"),
                   (67, "Figure5_stacking_external.png"),
                   (72, "Figure6_SHAP.png")]:
    replace_picture(P[pno], FIG_DIR / fname)
    print(f"[WORD] P[{pno}] <- {fname}")

# --- Figure 1 legend rewritten to match the three-column flowchart ----------
p44 = P[44]
assert p44.runs and p44.runs[0].text.startswith("Figure 1."), "P44 head run"
new_body = (
    " Flowchart of the study. The left panel shows patient selection from the "
    "MIMIC-IV database with the numbers retained at each step and the locked "
    "stratified 70/30 split; the middle panel shows the modelling workflow, in "
    "which univariable screening and LASSO selection were performed on the "
    "MIMIC-IV training set only and were re-run inside every cross-validation "
    "fold, base learners were ranked by training-set out-of-fold AUC and locked "
    "before the internal test set was opened, and the internal test set and the "
    "eICU-CRD cohort were examined only after the frozen final predictor had "
    "been fixed; the right panel shows the SHAP-based interpretability analyses "
    "performed on the frozen predictor.")
for r in p44.runs[1:]:
    r._element.getparent().remove(r._element)
p44.add_run(new_body)
print("[WORD] P[44] Figure 1 legend rewritten")
d.save(str(DOCX))
print("[SAVED]", DOCX.name)

# ================================================================ 2. PPT ====
prs = Presentation(str(DECK))
S = list(prs.slides)


def replace_in_tf(tf, old, new):
    hits = 0
    for para in tf.paragraphs:
        for r in para.runs:
            if old in r.text:
                r.text = r.text.replace(old, new)
                hits += 1
    return hits


edits = [
    (0,  "Figure 1  ·  Study flowchart",
         "Figure 1  ·  Flowchart of the study"),
    (7,  "Figure 3  ·  Nine candidate machine-learning models — ROC & PR, "
         "training set (OOF) vs internal test set.",
         "Figure 3  ·  Performance of the nine candidate machine-learning "
         "models — ROC & PR, training set (OOF) vs internal test set."),
    (12, "Figure 4 E / I", "Figure 4E / 5D"),
    (14, "Figure 6 — Internal test set (MIMIC-IV): ROC / PR / DCA",
         "Figure 4 B–D  ·  Internal test set (MIMIC-IV): ROC / PR / DCA"),
    (14, "Figure 6 — internal hold-out test set.",
         "Figure 4 (panels B–D) — internal hold-out test set."),
    (15, "Figure 7 — External validation (eICU-CRD): ROC / PR / DCA",
         "Figure 5 A–C  ·  External validation (eICU-CRD): ROC / PR / DCA"),
    (15, "Figure 7 — eICU-CRD external validation.",
         "Figure 5 (panels A–C) — eICU-CRD external validation."),
    (16, "Figure 8A", "Figure 6A"),
    (17, "Figure 8B", "Figure 6B"),
    (18, "Figure 8C", "Figure 6C"),
    (19, "Figure 8D", "Figure 6D"),
    (20, "Figure 8E", "Figure 6E"),
    (21, "Figure 8F", "Figure 6F"),
]
for si, old, new in edits:
    n = 0
    for sh in S[si].shapes:
        if sh.has_text_frame:
            n += replace_in_tf(sh.text_frame, old, new)
    print(f"[PPT] slide {si+1}: {old[:42]!r} -> {new[:42]!r}  ({n} run(s))")
    assert n >= 1, f"no hit on slide {si+1}: {old!r}"

prs.save(str(DECK))
print("[SAVED] deck")
