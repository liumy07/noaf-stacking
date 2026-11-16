# -*- coding: utf-8 -*-
"""step74_fix_age_ci.py — correct the univariable age 95% CI in the manuscript.

The raw estimate (refit: OR 1.07253, 95% CI 1.06972-1.07534) rounds to
1.070-1.075, but both manuscript copies carried 1.071-1.075.

Applies the fix to
  NOAF_stacking_manuscript.docx
  NOAF_stacking_manuscript_EndNote_revised_v90.docx
verifies the EndNote field count is unchanged, and re-checks the value.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import docx
from docx.oxml.ns import qn

PROJ = Path(r"D:/BJND-4002-11项目/BJND-4002-11")
TARGETS = [PROJ / "NOAF_stacking_manuscript.docx",
           PROJ / "NOAF_stacking_manuscript_EndNote_revised_v90.docx"]

OLD, NEW = "1.071-1.075", "1.070-1.075"


def fields(d):
    body = d.element.body
    b = sum(1 for e in body.iter(qn("w:fldChar"))
            if e.get(qn("w:fldCharType")) == "begin")
    e_ = sum(1 for e in body.iter(qn("w:fldChar"))
             if e.get(qn("w:fldCharType")) == "end")
    instr = len(list(body.iter(qn("w:instrText"))))
    return b, e_, instr


for path in TARGETS:
    if not path.exists():
        print(f"[skip] {path.name} missing")
        continue
    d = docx.Document(str(path))
    before = fields(d)
    hits = 0
    for p in d.paragraphs:
        if OLD not in p.text:
            continue
        for r in p.runs:
            if OLD in r.text:
                r.text = r.text.replace(OLD, NEW)
                hits += 1
    if hits == 0:
        print(f"[skip] {path.name}: pattern not found")
        continue

    tmp = path.with_name("_step74_tmp.docx")
    d.save(str(tmp))
    d2 = docx.Document(str(tmp))  # reopen to validate
    after = fields(d2)
    txt = "\n".join(p.text for p in d2.paragraphs)
    ok_val = NEW in txt and OLD not in txt
    print(f"{path.name}: runs patched {hits} | fields {before} -> {after} | value ok {ok_val}")
    if not ok_val or before != after:
        print("  [ABORT] verification failed, original left untouched")
        tmp.unlink(missing_ok=True)
        continue
    for i in range(8):
        try:
            shutil.copyfile(tmp, path)
            print(f"  [OK] written")
            break
        except PermissionError:
            print(f"  locked, retry {i}")
            time.sleep(3)
    else:
        print("  [FAIL] file locked; patched copy kept at", tmp)
        continue
    tmp.unlink(missing_ok=True)
