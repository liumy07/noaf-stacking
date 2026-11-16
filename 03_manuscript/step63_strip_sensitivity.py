# -*- coding: utf-8 -*-
"""Delete the sensitivity-analysis narrative from the Methods paragraph and TRIPOD checklist."""
import os
import shutil
import sys

from docx import Document

MS = r"D:/BJND-4002-11项目/BJND-4002-11/NOAF_stacking_manuscript_EndNote_revised_v90.docx"
TRIPOD = r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/rev_noaf/TRIPOD+AI_checklist_NOAF_stacking.docx"

# ---------------- manuscript ----------------
doc = Document(MS)
p = doc.paragraphs[21]
runs = p.runs
assert len(runs) == 8, len(runs)
tgt = runs[6]
assert "in contrast to our original analysis" in tgt.text
assert "As a sensitivity analysis" in tgt.text
assert "fldChar" not in tgt._r.xml and "instrText" not in tgt._r.xml

OLD_CLAUSE = ("); in contrast to our original analysis, in which imputation was applied to each cohort "
              "in full before any split, the imputation stage was moved inside the analysis pipeline:")
NEW_CLAUSE = "). Imputation was performed inside the analysis pipeline:"

OLD_SENS = (" As a sensitivity analysis, the imputer was instead fitted once on the full training set "
            "before cross-validation and the whole development loop repeated; discrimination was essentially "
            "unchanged (cross-validated AUC of the stacking ensemble 0.8416 vs. 0.8418, internal test set "
            "0.8429 vs. 0.8430, eICU-CRD 0.7526 vs. 0.7531, and no single model changed by more than 0.001 "
            "in any evaluation set; Supplementary Table S10), so the pipeline reported here is the fully "
            "leakage-free one.")

txt = tgt.text
assert txt.count(OLD_CLAUSE) == 1
assert txt.count(OLD_SENS) == 1
txt = txt.replace(OLD_CLAUSE, NEW_CLAUSE).replace(OLD_SENS, "")
tgt.text = txt

p2 = doc.paragraphs[21]
assert "original analysis" not in p2.text
assert "sensitivity analysis" not in p2.text
assert "S10" not in p2.text
assert "preprocessing included. Because a single completed dataset" in p2.text

tmp = MS + ".tmp"
doc.save(tmp)
try:
    shutil.copyfile(tmp, MS)
    os.remove(tmp)
    print("manuscript saved in place")
except PermissionError:
    alt = MS.replace("_v90.docx", "_v90_clean.docx")
    shutil.copyfile(tmp, alt)
    os.remove(tmp)
    print("manuscript locked -> saved to", alt)
    sys.exit(2)

# ---------------- TRIPOD ----------------
td = Document(TRIPOD)
cell = td.tables[0].rows[20].cells[4]
run = cell.paragraphs[0].runs[0]
OLD_T = "; fold-internal refitting sensitivity analysis in Supplementary Table S6"
assert OLD_T in run.text
run.text = run.text.replace(OLD_T, "")
td.save(TRIPOD)
print("tripod saved:", cell.text)
