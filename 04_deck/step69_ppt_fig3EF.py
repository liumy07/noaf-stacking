# -*- coding: utf-8 -*-
"""
step69_ppt_fig3EF.py — deck-side changes for the external nine-model curves.

1. Slide 8 title: "Figure 3  ·" -> "Figure 3A–D  ·" (the manuscript's
   Figure 3 is now a 3x2 panel matrix; the deck splits it like it already
   splits Figures 4/5).
2. Slide 8 notes: point to the new E-F slide.
3. NEW slide inserted directly after slide 8:
      title  : Figure 3E–F · external eICU-CRD ROC & PR
      picture: Fig3EF_External_ROC_PR.png (aspect 1.609)
      notes  : full caption with the Table 2 numbers
   The slide clones slide 8's title textbox (same font, size, position)
   and is moved into position 9 of the sldIdLst.
"""
import copy
import shutil
import time
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches
from pptx.oxml.ns import qn

DECK = Path(r"<LOCAL_ROOT>/Desktop/A prediction model for new-onset "
            r"atrial fibrillation in critically ill patients based on stacking "
            r"ensemble strategy/新结果figure设计_修订版.pptx")
TMP = DECK.with_name("_deck_tmp.pptx")
PNG = Path(r"<LOCAL_ROOT>/WorkBuddy/2026-09-09-17-02-31/output/"
           r"rev_noaf/v90_primary/figures/Fig3EF_External_ROC_PR.png")
PNG_AR = 4582 / 2847  # width / height

prs = Presentation(str(DECK))
s8 = prs.slides[7]

# -- 1. retitle slide 8 ------------------------------------------------------
done = False
for sh in s8.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Figure 3"):
        para = sh.text_frame.paragraphs[0]
        r0 = para.runs[0]
        assert "Figure 3  ·" in r0.text, r0.text[:40]
        r0.text = r0.text.replace("Figure 3  ·", "Figure 3A–D  ·", 1)
        done = True
        print("[S8 title]", sh.text_frame.text[:80])
        break
assert done

# -- 2. slide 8 notes --------------------------------------------------------
ntf = s8.notes_slide.notes_text_frame
tail = " Panels E and F (external eICU-CRD cohort) are shown on the next slide."
if tail not in ntf.text:
    ntf.text = ntf.text + tail
print("[S8 notes]", ntf.text[-90:])

# -- 3. new slide ------------------------------------------------------------
layout = s8.slide_layout
new = prs.slides.add_slide(layout)

# clone the title textbox from slide 8 (plain textbox: no rels involved)
title_src = next(sh for sh in s8.shapes
                 if sh.has_text_frame and sh.text_frame.text.startswith("Figure 3A–D"))
el = copy.deepcopy(title_src._element)
new.shapes._spTree.append(el)

new_title = None
for sh in new.shapes:
    if sh.has_text_frame:
        sh.width = Inches(12.40)
        sh.left = Inches(0.45)
        new_title = sh
assert new_title is not None
para = new_title.text_frame.paragraphs[0]
para.runs[0].text = ("Figure 3E–F  ·  External validation of the nine candidate "
                     "machine-learning models — ROC & PR, eICU-CRD cohort "
                     "(n = 32,067). CIs in the legends.")
for r in para.runs[1:]:
    r.text = ""
print("[NEW title]", new_title.text_frame.text[:80])

# picture: keep the PNG aspect, fit the vertical space below the title
h = 6.78
w = h * PNG_AR
pic = new.shapes.add_picture(str(PNG), Inches((13.3333 - w) / 2),
                             Inches(0.56), height=Inches(h))
print(f"[NEW pic] {pic.width/914400:.2f} x {pic.height/914400:.2f} in")

notes = new.notes_slide.notes_text_frame
notes.text = (
    "Figure 3E–F — External validation of the nine candidate machine-learning "
    "models in the eICU-CRD cohort (n = 32,067; NOAF prevalence 7.7%). "
    "(E) ROC curves: XGBoost 0.751 (95% CI 0.742–0.761) and LightGBM 0.750 "
    "(0.740–0.760) highest, followed by the RF 0.746 (0.736–0.755) and the "
    "MLP 0.744 (0.735–0.754); KNN lowest at 0.696 (0.686–0.706). "
    "(F) Precision-recall curves: highest average precision XGBoost 0.216 "
    "(95% CI 0.204–0.229) and LightGBM 0.216 (0.204–0.229), against a "
    "prevalence of 7.7%. Conventional logistic regression remained "
    "competitive (AUC 0.738, 95% CI 0.728–0.747). AUCs carry DeLong 95% CIs "
    "and APs bootstrap 95% CIs (1,000 resamples); every numeric estimate for "
    "every model and cohort is given in Table 2. Panels A–D of this figure "
    "are on the previous slide.")

# -- 4. move the new slide to position 9 (index 8) ---------------------------
lst = prs.slides._sldIdLst
ids = list(lst)
new_id = ids[-1]
lst.remove(new_id)
lst.insert(8, new_id)

prs.save(str(TMP))
print("[SAVED]", TMP, "slides:", len(Presentation(str(TMP)).slides.__iter__.__self__._sldIdLst if False else Presentation(str(TMP)).slides._sldIdLst))

# -- 5. replace the original (retry on lock) ---------------------------------
for attempt in range(6):
    try:
        shutil.copyfile(TMP, DECK)
        print("[OK] deck overwritten")
        break
    except PermissionError:
        time.sleep(3)
else:
    print("[FAIL] deck locked; kept", TMP.name)
