# -*- coding: utf-8 -*-
"""
step56b_fix_manuscript_v90.py — redo the two v90 manuscript edits WITHOUT
destroying EndNote fields.

Why this exists: step56 used `runs[0].text = new_full` + `r.text = ""` on every
other run.  python-docx's Run.text setter calls clear_content(), which wipes
*all* child elements of the run — including <w:fldChar> and <w:instrText>.
That silently deleted the P21 EndNote citation (43 -> 42 ADDIN fields).

This version starts from the untouched pre-v90 backup and edits ONLY the single
text-only run that actually contains each anchor (verified: no field chars, no
w:br / w:tab inside), leaving every field run byte-identical.
"""
from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

import pandas as pd
from docx import Document
from lxml import etree

REV_DIR = Path(r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/rev_noaf")
BASE = Path(r"D:/BJND-4002-11项目/BJND-4002-11")
SRC = BASE / "NOAF_stacking_manuscript_EndNote_revised_pre-v90.docx"
OUT_MAIN = BASE / "NOAF_stacking_manuscript_EndNote_revised.docx"
OUT_ALT = BASE / "NOAF_stacking_manuscript_EndNote_revised_v90.docx"

# ------------------------------------------------------------------
# numbers straight from the step55 comparison output
# ------------------------------------------------------------------
df = pd.read_csv(REV_DIR / "v90_fold_impute" / "v90_sensitivity_AUC.csv")
dl = pd.read_csv(REV_DIR / "v90_fold_impute" / "v90_sensitivity_delong.csv")

max_abs = float(df["delta"].abs().max())
stack_test = df[(df.Model == "Stacking") & (df.cohort == "internal test")].iloc[0]
stack_eicu = df[(df.Model == "Stacking") & (df.cohort == "eICU external")].iloc[0]
stack_oof = df[(df.Model == "Stacking") & (df.cohort == "train OOF")].iloc[0]
dl_test = dl[dl.cohort == "internal test"].iloc[0]
dl_eicu = dl[dl.cohort == "eICU external"].iloc[0]

sens_clause = (
    f"the imputer was refitted inside every cross-validation fold on that "
    f"fold's training rows and the whole pipeline (screening, the nine "
    f"learners, meta-learner training and model ranking) was repeated. This "
    f"left discrimination essentially unchanged: the cross-validated AUC of "
    f"the stacking ensemble moved from {stack_oof.AUC_v80:.4f} to "
    f"{stack_oof.AUC_v90:.4f}, its AUC in the internal test set from "
    f"{stack_test.AUC_v80:.4f} to {stack_test.AUC_v90:.4f} and in eICU-CRD "
    f"from {stack_eicu.AUC_v80:.4f} to {stack_eicu.AUC_v90:.4f}, and no single "
    f"model changed by more than "
    f"{max(0.001, math.ceil(max_abs * 1000) / 1000):.3f} in any evaluation set "
    f"(Supplementary Table S10); the frozen ensemble applied to the two "
    f"evaluation cohorts is in any case unchanged in construction, because the "
    f"imputer is fitted on training rows only and never sees the evaluation "
    f"cohorts"
)

# ------------------------------------------------------------------
# field-safe single-run replacement
# ------------------------------------------------------------------
doc = Document(SRC)


def replace_in_run(p, old: str, new: str) -> None:
    """Replace `old` by `new` inside the ONE text-only run that holds it.

    Raises if the span is split across runs or touches a field run — better to
    stop than to risk another silent citation loss.
    """
    full = "".join(r.text for r in p.runs)
    if full.count(old) != 1:
        raise SystemExit(f"anchor not unique/found (count={full.count(old)}): {old[:60]}...")

    start = full.find(old)
    end = start + len(old)
    pos = 0
    hit = []
    for r in p.runs:
        s, e = pos, pos + len(r.text)
        if s < end and e > start:
            hit.append((r, s, e))
        pos = e

    if len(hit) != 1:
        raise SystemExit(f"anchor spans {len(hit)} runs — refusing to edit")

    r, s, e = hit[0]
    xml = r._r.xml
    if "ADDIN" in xml or "fldChar" in xml or "instrText" in xml:
        raise SystemExit("anchor run contains a field — refusing to edit")
    if "<w:br" in xml or "<w:tab" in xml or "<w:drawing" in xml:
        raise SystemExit("anchor run contains br/tab/drawing — refusing to edit")

    t = r.text
    r.text = t[: start - s] + new + t[end - s:]


# --- edit 0: qualify the primary-analysis statement ----------------------
replace_in_run(doc.paragraphs[21],
               "Imputation was not refitted inside the cross-validation folds.",
               "In the primary analysis, imputation was not refitted inside "
               "the cross-validation folds.")
print("[OK] P21 primary-analysis qualifier added")

# --- edit 1: Methods 2.3 -------------------------------------------------
anchor21 = ("the leakage protections that matter for the reported performance "
            "- variable screening, model fitting, model ranking and evaluation "
            "- were enforced without exception.")
new21 = anchor21 + (" As a sensitivity analysis, " + sens_clause + "; the "
                    "published pipeline is therefore retained as the primary "
                    "analysis.")
replace_in_run(doc.paragraphs[21], anchor21, new21)
print("[OK] Methods 2.3 sensitivity sentence inserted")

# --- edit 2: Limitations -------------------------------------------------
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
replace_in_run(doc.paragraphs[84], anchor84, new84)
print("[OK] Limitations external-calibration explanation inserted")

# ------------------------------------------------------------------
# save (main file may be locked by Word / preview)
# ------------------------------------------------------------------
tmp = BASE / "~v90b_tmp.docx"
doc.save(tmp)
saved_main = False
for _ in range(3):
    try:
        os.replace(tmp, OUT_MAIN)
        saved_main = True
        break
    except PermissionError:
        time.sleep(3)
if not saved_main:
    shutil.copy2(tmp, OUT_ALT)
    os.remove(tmp)
    print(f"[LOCKED] original open elsewhere; edits written to {OUT_ALT.name}")
else:
    if OUT_ALT.exists():
        OUT_ALT.unlink()
    print(f"[SAVED] {OUT_MAIN.name}")

# ------------------------------------------------------------------
# verify
# ------------------------------------------------------------------
target = OUT_MAIN if saved_main else OUT_ALT
a = Document(SRC)
b = Document(target)


def n_addin(d):
    return etree.tostring(d.element.body, encoding="unicode").count("ADDIN")


print(f"[VERIFY] ADDIN fields: before={n_addin(a)}  after={n_addin(b)}")
assert n_addin(a) == n_addin(b), "EndNote field count changed!"

t21, t84 = b.paragraphs[21].text, b.paragraphs[84].text
assert "Supplementary Table S10" in t21 and "sensitivity analysis" in t21
assert "7.7% vs. 27.7%" in t84
print("[VERIFY] both edits present, all EndNote fields intact")

# no other paragraph changed
changed = [i for i, (pa, pb) in enumerate(zip(a.paragraphs, b.paragraphs))
           if pa.text != pb.text]
print(f"[VERIFY] changed paragraphs: {changed}")
