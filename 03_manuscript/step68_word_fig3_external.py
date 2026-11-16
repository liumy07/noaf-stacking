# -*- coding: utf-8 -*-
"""
step68_word_fig3_external.py

Word-side changes for the external nine-model curves (Figure 3E-F):

1. Heading 3.3  -> "3.3  Development and validation of nine candidate
   machine-learning models"  (drops the ambiguous "internal-external"
   compound; the section covers both validation cohorts and the body text
   says so explicitly).
2. Para 56 first sentence: development happens on the MIMIC-IV training set;
   only the benchmarking spans both cohorts.
3. Para 57: cite Figure 3E (external ROC) and Figure 3F (external PR).
4. Para 58: replace the Figure 3 image with the 3x2 composite
   Fig3_NineModels_AllCohorts.png (A/B OOF, C/D internal test, E/F external),
   width kept at 446.4 pt, height from the PNG aspect ratio.
5. Para 59 caption: add the (E)/(F) clause.

EndNote fields are untouched (all edits are plain-text runs verified to be
field-free).  Main file and the _v90 sync copy are both updated.
"""
import shutil
import time
from pathlib import Path

import docx
from docx.oxml.ns import qn
# PNG size of Fig3_NineModels_AllCohorts.png (read once via PIL elsewhere)
PNG_W, PNG_H = 4582, 5882

MAIN = Path(r"D:/BJND-4002-11项目/BJND-4002-11/NOAF_stacking_manuscript_EndNote_revised.docx")
SYNC = Path(r"D:/BJND-4002-11项目/BJND-4002-11/NOAF_stacking_manuscript_EndNote_revised_v90.docx")
TMP = Path(r"D:/BJND-4002-11项目/BJND-4002-11/_tmp_main_fixed.docx")
PNG = Path(r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/rev_noaf/v90_primary/figures/Fig3_NineModels_AllCohorts.png")

WIDTH_PT = 446.4


def count_fields(d):
    xml = d.element.xml
    return (xml.count('w:fldCharType="begin"'),
            xml.count('w:fldCharType="end"'),
            xml.count("<w:instrText"))


def main():
    d = docx.Document(str(MAIN))
    ps = d.paragraphs
    b0, e0, i0 = count_fields(d)
    print(f"[fields before] begin={b0} end={e0} instr={i0}")

    # -- 1. heading 3.3 ----------------------------------------------------
    r = ps[55].runs
    assert r[0].text.startswith("3.3"), r[0].text
    r[0].text = "3.3  Development"
    r[1].text = " and "
    r[2].text = ""
    r[3].text = "validation of nine candidate machine-learning models"
    print("[H 3.3]", ps[55].text)

    # -- 2. para 56 first sentence -----------------------------------------
    p56 = ps[56].runs[0]
    old = "We then developed and benchmarked the nine candidate classification algorithms across both cohorts."
    new = ("We then developed the nine candidate classification algorithms on "
           "the MIMIC-IV training set and benchmarked them in both validation "
           "cohorts.")
    assert old in p56.text, p56.text[:120]
    p56.text = p56.text.replace(old, new)
    print("[P56]", ps[56].text[:150])

    # -- 3. para 57 citations ----------------------------------------------
    p57 = ps[57].runs[0]
    t = p57.text
    a = "(0.696, 95% CI 0.686-0.706; Table 2)"
    assert a in t
    t = t.replace(a, "(0.696, 95% CI 0.686-0.706; Figure 3E; Table 2)")
    b = "against a prevalence of 0.077."
    assert b in t
    t = t.replace(b, "against a prevalence of 0.077 (Figure 3F).")
    p57.text = t
    print("[P57] citations inserted")

    # -- 4. swap Figure 3 image --------------------------------------------
    w, h = PNG_W, PNG_H
    cx = int(WIDTH_PT / 72 * 914400)
    cy = int(WIDTH_PT * h / w / 72 * 914400)
    run = ps[58].runs[0]
    blip = run._r.findall(".//" + qn("a:blip"))[0]
    old_rid = blip.get(qn("r:embed"))
    new_rid, _img = d.part.get_or_add_image(str(PNG))
    blip.set(qn("r:embed"), new_rid)
    for tag in (qn("wp:extent"),):
        for el in ps[58]._p.findall(".//" + tag):
            el.set("cx", str(cx)); el.set("cy", str(cy))
    for el in ps[58]._p.findall(".//" + qn("a:ext")):
        el.set("cx", str(cx)); el.set("cy", str(cy))
    print(f"[IMG] {old_rid} -> {new_rid}  {cx/914400*72:.1f} x {cy/914400*72:.1f} pt")

    # drop the old image relationship if no longer referenced
    still = any(b.get(qn("r:embed")) == old_rid
                for b in d.element.body.findall(".//" + qn("a:blip")))
    if not still and old_rid in d.part.rels:
        d.part.drop_rel(old_rid)
        print("[IMG] old rel dropped")

    # -- 5. caption ---------------------------------------------------------
    cap = ps[59].runs[0]
    t = cap.text
    c = "(D) precision-recall curves on the MIMIC-IV internal test set (n = 10,316)."
    assert c in t, t[:200]
    t = t.replace(c, "(D) precision-recall curves on the MIMIC-IV internal "
                     "test set (n = 10,316); (E) ROC curves and (F) "
                     "precision-recall curves in the external eICU-CRD "
                     "cohort (n = 32,067).")
    cap.text = t
    print("[CAP]", ps[59].text[:260])

    b1, e1, i1 = count_fields(d)
    print(f"[fields after] begin={b1} end={e1} instr={i1}")
    assert (b0, e0, i0) == (b1, e1, i1), "EndNote field count changed!"

    d.save(str(TMP))
    print("[SAVED]", TMP)

    # -- 6. copy back over the locked originals ----------------------------
    for target in (MAIN, SYNC):
        for attempt in range(6):
            try:
                shutil.copyfile(TMP, target)
                print("[OK]", target.name)
                break
            except PermissionError:
                time.sleep(3)
        else:
            raise RuntimeError(f"locked: {target}")


if __name__ == "__main__":
    main()
