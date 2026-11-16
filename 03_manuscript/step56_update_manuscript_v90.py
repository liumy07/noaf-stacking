# -*- coding: utf-8 -*-
"""
step56_update_manuscript_v90.py — insert the fold-internal-imputation
sensitivity analysis into the manuscript (two surgical edits, everything else
untouched):

  1. Methods 2.3 (missing-data paragraph): one sentence declaring the
     sensitivity analysis and its headline result (Supplementary Table S6).
  2. Discussion, limitations paragraph: explanation of WHY external
     calibration was poor (prevalence 7.7% vs 27.7%, case mix, single-centre
     vs multi-centre ascertainment/practice patterns) while discrimination
     transported.

Numbers are read from the step55 comparison output so the text can never
drift from the computed results.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document

REV_DIR = Path(r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/rev_noaf")
DOCX = Path(r"D:/BJND-4002-11项目/BJND-4002-11/NOAF_stacking_manuscript_EndNote_revised.docx")

# ------------------------------------------------------------------
# 1. numbers from the sensitivity comparison
# ------------------------------------------------------------------
df = pd.read_csv(REV_DIR / "v90_fold_impute" / "v90_sensitivity_AUC.csv")
dl = pd.read_csv(REV_DIR / "v90_fold_impute" / "v90_sensitivity_delong.csv")

max_abs = float(df["delta"].abs().max())          # largest |delta| over all
stack_test = df[(df.Model == "Stacking") & (df.cohort == "internal test")].iloc[0]
stack_eicu = df[(df.Model == "Stacking") & (df.cohort == "eICU external")].iloc[0]
stack_oof = df[(df.Model == "Stacking") & (df.cohort == "train OOF")].iloc[0]
dl_test = dl[dl.cohort == "internal test"].iloc[0]
dl_eicu = dl[dl.cohort == "eICU external"].iloc[0]

print(f"max |delta| over all models/cohorts: {max_abs:+.4f}")
print(f"Stacking OOF   : v80 {stack_oof.AUC_v80:.4f} -> v90 {stack_oof.AUC_v90:.4f}")
print(f"Stacking test  : v80 {stack_test.AUC_v80:.4f} -> v90 {stack_test.AUC_v90:.4f}")
print(f"Stacking eICU  : v80 {stack_eicu.AUC_v80:.4f} -> v90 {stack_eicu.AUC_v90:.4f}")
print(f"DeLong test/eICU P: {dl_test.p:.3f} / {dl_eicu.p:.3f}")

import math

sens_clause = (
    f"repeating the entire development pipeline with the imputer refitted "
    f"inside every cross-validation fold on that fold's training rows left the "
    f"discrimination of the stacking ensemble essentially unchanged (training "
    f"out-of-fold AUC {stack_oof.AUC_v90:.3f} vs. {stack_oof.AUC_v80:.3f}; "
    f"internal test {stack_test.AUC_v90:.3f} vs. {stack_test.AUC_v80:.3f}; "
    f"eICU-CRD {stack_eicu.AUC_v90:.3f} vs. {stack_eicu.AUC_v80:.3f}; largest "
    f"absolute change {max(abs(stack_oof.delta), abs(stack_test.delta), abs(stack_eicu.delta)):.4f}), "
    f"and no single model changed by more than "
    f"{max(0.001, math.ceil(max_abs * 1000) / 1000):.3f} in any evaluation set "
    f"(Supplementary Table S6)")

# ------------------------------------------------------------------
# 2. surgical edits
# ------------------------------------------------------------------
doc = Document(DOCX)


def replace_in_paragraph(p, old: str, new: str) -> bool:
    """Replace the first occurrence of `old` in the paragraph, preserving the
    first run's formatting. Fails loudly if `old` is absent or not unique."""
    runs = p.runs
    full = "".join(r.text for r in runs)
    if full.count(old) != 1:
        return False
    new_full = full.replace(old, new, 1)
    runs[0].text = new_full
    for r in runs[1:]:
        r.text = ""
    return True


# --- edit 1: Methods 2.3 — declare the sensitivity analysis ------------
p21 = doc.paragraphs[21]
anchor21 = ("the leakage protections that matter for the reported performance "
            "- variable screening, model fitting, model ranking and evaluation "
            "- were enforced without exception.")
new21 = anchor21 + (" As a sensitivity analysis, " + sens_clause + "; the "
                    "published pipeline is retained as the primary analysis "
                    "because the imputer is an outcome-blind, unsupervised "
                    "transform.")
if not replace_in_paragraph(p21, anchor21, new21):
    raise SystemExit("EDIT 1 FAILED: anchor not found in P21")
print("[OK] Methods 2.3 sensitivity sentence inserted")

# --- edit 2: Limitations — explain the poor external calibration --------
p84 = doc.paragraphs[84]
anchor84 = ("because calibration in the external cohort was poor and the "
            "model would require local recalibration before use.")
new84 = ("because calibration in the external cohort was poor and the model "
         "would require local recalibration before use; this miscalibration is "
         "plausibly attributable to the much lower event prevalence in "
         "eICU-CRD (7.7% vs. 27.7%), to differences in case mix and in how "
         "rhythm disturbances were ascertained between a single tertiary "
         "referral centre and a multi-centre registry, and to inter-institution "
         "variation in documentation and treatment practice, rather than to a "
         "loss of discriminative information, which the comparable ordering of "
         "the models across cohorts supports.")
if not replace_in_paragraph(p84, anchor84, new84):
    raise SystemExit("EDIT 2 FAILED: anchor not found in P84")
print("[OK] Limitations external-calibration explanation inserted")

# backup first, then save (the file may be open in Word / the preview panel)
import shutil
import os
import time

bak = DOCX.with_name("NOAF_stacking_manuscript_EndNote_revised_pre-v90.docx")
shutil.copy2(DOCX, bak)
tmp = DOCX.with_name("~v90_tmp.docx")
doc.save(tmp)
for _ in range(6):
    try:
        os.replace(tmp, DOCX)
        print(f"[SAVED] {DOCX}")
        break
    except PermissionError:
        time.sleep(5)
else:
    alt = DOCX.with_name("NOAF_stacking_manuscript_EndNote_revised_v90.docx")
    shutil.copy2(tmp, alt)
    os.remove(tmp)
    print(f"[LOCKED] original is open; edits saved to {alt}")
    DOCX = alt
print(f"[BACKUP] {bak}")

# ------------------------------------------------------------------
# 3. verify: no placeholder text, anchors replaced, EndNote fields intact
# ------------------------------------------------------------------
import re
from lxml import etree
doc2 = Document(DOCX)
t21 = doc2.paragraphs[21].text
t84 = doc2.paragraphs[84].text
assert "sensitivity analysis" in t21 and "Supplementary Table S6" in t21
assert "7.7% vs. 27.7%" in t84
xml = etree.tostring(doc2.element.body, encoding="unicode")
assert "ADDIN" in xml, "EndNote fields lost!"
n_addin = xml.count("ADDIN")
print(f"[VERIFY] EndNote ADDIN fields: {n_addin}")
print("[VERIFY] both edits present")
