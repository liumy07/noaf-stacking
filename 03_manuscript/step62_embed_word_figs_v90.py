# -*- coding: utf-8 -*-
"""step62_embed_word_figs_v90.py

Re-embed the six manuscript figures from the v90 run into the manuscript,
keeping each picture's width and its caption paragraph untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Emu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import FIG_DIR  # noqa: E402  (NOAF_RUN=v90 -> v90_primary)

DOCX = Path(r"D:/BJND-4002-11项目/BJND-4002-11/NOAF_stacking_manuscript_EndNote_revised_v90.docx")

MAP = [(43, "Figure_Flowchart_study.png"),
       (53, "Fig2_Screening.png"),
       (58, "Fig3_NineModels_TrainTest.png"),
       (65, "Figure4_stacking_internal.png"),
       (67, "Figure5_stacking_external.png"),
       (72, "Figure6_SHAP.png")]


def fig_width(p):
    ext = p._element.findall(".//" + qn("wp:extent"))
    return int(ext[0].get("cx")) if ext else Emu(int(15.5 * 360000))


def replace_picture(p, img):
    w = fig_width(p)
    for r in list(p.runs):
        if r._element.findall(".//" + qn("w:drawing")):
            r._element.getparent().remove(r._element)
    p.add_run().add_picture(str(img), width=Emu(w))


def main():
    doc = Document(str(DOCX))
    P = doc.paragraphs
    for pno, fname in MAP:
        img = FIG_DIR / fname
        if not img.exists():
            print(f"[SKIP] {fname} not available")
            continue
        replace_picture(P[pno], img)
        print(f"[WORD] P{pno} <- {fname}")
    doc.save(str(DOCX))
    print("[SAVED]", DOCX.name)


if __name__ == "__main__":
    main()
