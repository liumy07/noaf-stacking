# -*- coding: utf-8 -*-
"""step52b_resync_pictures.py — idempotent picture sync (no caption edits):
embed the refreshed v80 figures into the Word manuscript and the PPT slides
that show the same panels, after the letter re-mapping (step43 patch).
"""
from __future__ import annotations

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

# ------------------------------------------------------------------ WORD ---
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
    p.add_run().add_picture(str(img), width=Emu(w))


for pno, fname in [(43, "Figure_Flowchart_study.png"),
                   (53, "Fig2_Screening.png"),
                   (58, "Fig3_NineModels_TrainTest.png"),
                   (65, "Figure4_stacking_internal.png"),
                   (67, "Figure5_stacking_external.png"),
                   (72, "Figure6_SHAP.png")]:
    replace_picture(P[pno], FIG_DIR / fname)
d.save(str(DOCX))
print("[WORD] 6 pictures re-embedded")

# ------------------------------------------------------------------- PPT ---
prs = Presentation(str(DECK))
S = list(prs.slides)
ppt_map = {6: "Fig2_Screening.png",          # slide 7
           7: "Fig3_NineModels_TrainTest.png",  # slide 8
           9: "FigS1S_TrainOOF_ROC.png",     # slide 10 left pic
           12: "Fig4EI_calibration_two_cohorts.png",  # slide 13
           14: "Fig6_ROC_PR_DCA_internal.png",  # slide 15
           15: "Fig7_ROC_PR_DCA_eICU.png"}   # slide 16


def swap_first_pic(slide, img_path):
    pics = [sh for sh in slide.shapes if sh.__class__.__name__ == "Picture"]
    if not pics:
        return False
    pic = pics[0]
    part, rId = slide.part.get_or_add_image_part(str(img_path))
    blip = pic._element.blipFill.find(qn("a:blip"))
    blip.set(qn("r:embed"), rId)
    return True


# slide 10 has TWO pictures (ROC + PR): swap by order (left, right)
for si, fname in ppt_map.items():
    slide = S[si]
    if si == 9:
        pics = [sh for sh in slide.shapes if sh.__class__.__name__ == "Picture"]
        pics.sort(key=lambda s: s.left)
        for pic, fn in zip(pics, ["FigS1S_TrainOOF_ROC.png",
                                  "FigS1S_TrainOOF_PR.png"]):
            part, rId = slide.part.get_or_add_image_part(str(FIG_DIR / fn))
            pic._element.blipFill.find(qn("a:blip")).set(qn("r:embed"), rId)
        print(f"[PPT] slide {si+1}: 2 pictures swapped")
        continue
    assert swap_first_pic(slide, FIG_DIR / fname), f"no picture on slide {si+1}"
    print(f"[PPT] slide {si+1} <- {fname}")

prs.save(str(DECK))
print("[SAVED] deck")
