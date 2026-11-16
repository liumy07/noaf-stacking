# -*- coding: utf-8 -*-
"""Renumber figure references inside PowerPoint speaker notes to the current manuscript numbering."""
import re
import shutil

from pptx import Presentation

FN = "新结果figure设计_修订版.pptx"
OUT = "新结果figure设计_修订版_fix2.pptx"

RULES = [
    # cross-references
    ("Figures 6 and 7", "Figures 4B\u2013D and 5A\u2013C"),
    ("Figure 4 E / I", "Figure 4E / 5D"),
    ("Figure 4 E/I", "Figures 4E and 5D"),
    ("Supplementary Figure S4", "Supplementary Table S4"),
    # per-figure captions
    ("Figure 6 \u2014 Internal test set", "Figure 4B\u2013D \u2014 Internal test set"),
    ("Figure 6 \u2014 internal hold-out test set", "Figure 4B\u2013D \u2014 internal hold-out test set"),
    ("Figure 7 \u2014 External validation", "Figure 5A\u2013C \u2014 External validation"),
    ("Figure 7 \u2014 eICU-CRD external validation", "Figure 5A\u2013C \u2014 eICU-CRD external validation"),
    # SHAP figures 8A-8F -> 6A-6F
    ("Figure 8A", "Figure 6A"), ("Figure 8B", "Figure 6B"), ("Figure 8C", "Figure 6C"),
    ("Figure 8D", "Figure 6D"), ("Figure 8E", "Figure 6E"), ("Figure 8F", "Figure 6F"),
    # values refreshed to the leakage-free (v90) results
    ("AUC 0.846 (95% CI 0.838\u20130.854), AP 0.675 (95% CI 0.659\u20130.691)",
     "AUC 0.843 (95% CI 0.835\u20130.851), AP 0.668 (95% CI 0.651\u20130.684)"),
    ("at 20% +0.176 (95% CI +0.172 to +0.180)",
     "at 20% +0.174 (95% CI +0.170 to +0.178)"),
    ("AUC 0.755 (95% CI 0.746\u20130.765), AP 0.210 (95% CI 0.199\u20130.222)",
     "AUC 0.753 (95% CI 0.743\u20130.762), AP 0.201 (95% CI 0.190\u20130.213)"),
    ("at 10% +0.005 (95% CI +0.004 to +0.006)",
     "at 10% +0.004 (95% CI +0.003 to +0.005)"),
    ("mean 0.976", "mean 0.977 (min 0.969)"),
    ("(10 re-drawn explain sets, reduced protocol)", "(10 independent re-drawn explain sets)"),
    ("10 re-drawn explain sets (reduced protocol)",
     "10 independent re-drawn explain sets (same 300-patient protocol)"),
]

prs = Presentation(FN)
changed = []
for i, slide in enumerate(prs.slides, 1):
    if not slide.has_notes_slide:
        continue
    for para in slide.notes_slide.notes_text_frame.paragraphs:
        if not para.runs:
            continue
        old = "".join(r.text for r in para.runs)
        new = old
        for a, b in RULES:
            new = new.replace(a, b)
        if new != old:
            assert len(para.runs) == 1, (i, len(para.runs))
            para.runs[0].text = new
            changed.append(i)

print("slides changed:", sorted(set(changed)))

# post-check: no stale references left in notes
bad = []
for i, slide in enumerate(prs.slides, 1):
    if not slide.has_notes_slide:
        continue
    t = slide.notes_slide.notes_text_frame.text
    for m in re.finditer(r"Figure\s*\d+[A-F]?|Figures\s*\d+[^,\\.;]*", t):
        s = m.group(0)
        if re.match(r"Figure\s*8", s) or "Figure 6 \u2014" in t or "Figure 7 \u2014" in t:
            bad.append((i, s))
print("residual stale refs:", bad)

prs.save(OUT)
print("saved", OUT)

try:
    shutil.copyfile(OUT, FN)
    print("overwrote original")
except PermissionError:
    print("original locked -> kept", OUT)
