# -*- coding: utf-8 -*-
"""
step51_update_deck.py — replace every figure in 新结果figure设计_修订版.pptx
with the v80 results, update captions/tables, keep every picture + caption
editable and draggable (native picture shapes + text boxes).

Picture map (slide -> v80 figure), captions and tables updated to v80 numbers.
"""
import copy
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Emu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR  # noqa: E402

FIG = V80_DIR / "figures"
V70 = Path(__file__).resolve().parent / "v70_leakfree"
DECK = Path(r"C:\Users\localuser\Desktop\A prediction model for "
            r"new-onset atrial fibrillation in critically ill patients "
            r"based on stacking ensemble strategy\新结果figure设计_修订版.pptx")
BAK = Path(__file__).resolve().parent / "_backup_deck_修订版_pre_v80.pptx"

# ----------------------------------------------------------------- helpers
def f3(v):
    return f"{v:.3f}"


def ci(auc, m, coh, ap=False):
    df = AP if ap else AUC
    r = df[(df.Model == m) & (df.cohort == coh)].iloc[0]
    lo, hi = (r.lo, r.hi) if ap else (r.CI_lo, r.CI_hi)
    col = "AP" if ap else "AUC"
    return f"{r[col]:.3f} ({lo:.3f}\u2013{hi:.3f})"


def set_text(shape, text):
    """Replace text keeping the first run's formatting."""
    tf = shape.text_frame
    p = tf.paragraphs[0]
    # drop extra paragraphs
    for extra in list(tf.paragraphs)[1:]:
        extra._p.getparent().remove(extra._p)
    if p.runs:
        p.runs[0].text = text
        for r in list(p.runs)[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run(text)


def set_cell(cell, text):
    tf = cell.text_frame
    p = tf.paragraphs[0]
    for extra in list(tf.paragraphs)[1:]:
        extra._p.getparent().remove(extra._p)
    if p.runs:
        p.runs[0].text = text
        for r in list(p.runs)[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run(text)


def replace_pic(slide, pic, img_path):
    """Point the picture shape at a new image; keep width, recenter height."""
    img_part, rId = slide.part.get_or_add_image_part(str(img_path))
    blip = pic._element.blipFill.find(qn("a:blip"))
    blip.set(qn("r:embed"), rId)
    # drop any crop
    blipFill = pic._element.blipFill
    src = blipFill.find(qn("a:srcRect"))
    if src is not None:
        blipFill.remove(src)
    with Image.open(img_path) as im:
        w, h = im.size
    new_h = int(pic.width * h / w)
    top = pic.top + (pic.height - new_h) // 2
    pic.height = new_h
    pic.top = max(Emu(180000), top)


# ----------------------------------------------------------------- data
AUC = pd.read_csv(V80_DIR / "v80_ci_auc.csv")
AP = pd.read_csv(V80_DIR / "v80_ci_ap.csv")
RES = pd.read_csv(V80_DIR / "v80_results.csv")
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv")
V70M = pd.read_csv(V70 / "v70_nine_models_metrics.csv")

DECK_ORDER = ["MLP", "XGBoost", "RF", "LightGBM", "Logistic", "SVM",
              "Ridge", "DT", "KNN"]
NICE = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
        "RF": "Random forest", "LightGBM": "LightGBM",
        "Logistic": "Logistic regression", "SVM": "Support vector machine",
        "Ridge": "Ridge regression", "DT": "Decision tree",
        "KNN": "K-nearest neighbours", "Stacking": "Stacking ensemble"}

# ----------------------------------------------------------------- open
if not BAK.exists():
    shutil.copy2(DECK, BAK)
    print("[BACKUP]", BAK.name)
prs = Presentation(str(DECK))
S = list(prs.slides)

# ---------------------------------------------------------- 1. pictures
PICMAP = {
    (0, 0): "Figure_Flowchart_study",
    (6, 0): "Fig2_Screening",
    (7, 0): "Fig3_NineModels_TrainTest",
    (9, 0): "FigS1S_TrainOOF_ROC",
    (9, 1): "FigS1S_TrainOOF_PR",
    (10, 0): "Figure_Workflow_stacking",
    (12, 0): "Fig4EI_calibration_two_cohorts",
    (13, 0): "FigS4_delong_forest",
    (14, 0): "Fig6_ROC_PR_DCA_internal",
    (15, 0): "Fig7_ROC_PR_DCA_eICU",
    (16, 0): "Fig8_SHAP_beeswarm",
    (17, 0): "Fig8_SHAP_waterfall_TP",
    (18, 0): "Fig8_SHAP_waterfall_TN",
    (19, 0): "Fig8_SHAP_waterfall_FP",
    (20, 0): "Fig8_SHAP_waterfall_FN",
    (21, 0): "Fig8_SHAP_stability",
}
for (si, pi), stem in PICMAP.items():
    pics = [sh for sh in S[si].shapes
            if sh.__class__.__name__ == "Picture"]
    pics.sort(key=lambda p: (p.left, p.top))
    replace_pic(S[si], pics[pi], FIG / f"{stem}.png")
    print(f"[PIC] slide {si+1} -> {stem}")

# ---------------------------------------------------------- 2. captions
def shape_by_id(slide, sid):
    for sh in slide.shapes:
        if sh.shape_id == sid:
            return sh
    raise KeyError(sid)


set_text(shape_by_id(S[0], 4),
    "Study flowchart — split-first imputation; univariable screening and "
    "LASSO 1-SE re-run inside every cross-validation fold (26 variables "
    "retained); top-3 base learners locked by training-set out-of-fold AUC; "
    "one frozen predictor used for every evaluation and SHAP.")

set_text(shape_by_id(S[12], 4),
    "Calibration with bootstrap 95% CI band and intercept/slope 95% CI — "
    "internal calibration good (slope 1.14, 95% CI 1.09–1.19) though not "
    "perfect; eICU-CRD substantially miscalibrated (slope 0.72, intercept "
    "−1.73), recalibration required before absolute-risk use.")

set_text(shape_by_id(S[14], 4),
    "Figure 6 — internal hold-out test set. Stacking: AUC 0.843 (95% CI "
    "0.835–0.851), AP 0.668 (95% CI 0.652–0.684); net benefit at 10% +0.220 "
    "(95% CI +0.218 to +0.222) and at 20% +0.174 (95% CI +0.170 to +0.178). "
    "All AUC/AP values carry 95% CI in the legends; the DCA box carries "
    "bootstrap 95% CI.")

set_text(shape_by_id(S[15], 75),
    "Figure 7 — eICU-CRD external validation. Stacking: AUC 0.753 (95% CI "
    "0.744–0.763), AP 0.202 (95% CI 0.191–0.213); net benefit at 10% +0.004 "
    "(95% CI +0.003 to +0.005) and at 20% −0.025 (95% CI −0.027 to −0.024). "
    "All CIs shown in legends/box.")

set_text(shape_by_id(S[17], 114),
    "SHAP waterfall for an extreme high-confidence TRUE POSITIVE "
    "(pre-specified rule: highest predicted probability among correctly "
    "detected NOAF cases at the 0.5 threshold; predicted P(NOAF) = 0.913). "
    "Bars show each variable's additive contribution to P(NOAF) relative to "
    "the base value E[f(X)] = 0.317 estimated on the background sample.")

set_text(shape_by_id(S[18], 118),
    "SHAP waterfall for an extreme high-confidence TRUE NEGATIVE "
    "(pre-specified rule: lowest predicted probability among correctly "
    "excluded non-NOAF cases; predicted P(NOAF) = 0.062). Bars show each "
    "variable's additive contribution to P(NOAF) relative to the base value "
    "E[f(X)] = 0.317.")

set_text(shape_by_id(S[19], 122),
    "SHAP waterfall for an extreme high-confidence FALSE POSITIVE "
    "(pre-specified rule: highest predicted probability among non-NOAF cases "
    "flagged at the 0.5 threshold; predicted P(NOAF) = 0.910). The false "
    "alarm is driven mainly by the variables pushing the prediction above "
    "the base value.")

set_text(shape_by_id(S[20], 126),
    "SHAP waterfall for an extreme high-confidence FALSE NEGATIVE "
    "(pre-specified rule: lowest predicted probability among NOAF cases "
    "missed at the 0.5 threshold; predicted P(NOAF) = 0.063). "
    "Protective-direction variables outweighed the risk indicators for this "
    "patient.")

set_text(shape_by_id(S[21], 2),
    "Figure 8F — SHAP ranking stability (10 independent re-drawn explain "
    "sets)")
set_text(shape_by_id(S[21], 4),
    "Figure 8F — sampling stability of the SHAP feature ranking: 10 "
    "independent re-drawn explain sets (same 300-patient protocol); Spearman "
    "ρ vs the main analysis, mean 0.972 (min 0.957); top-10 Jaccard 0.806 — "
    "the ranking is stable.")
print("[CAPTIONS] updated")

# ---------------------------------------------------------- 3. tables
def table_by_idx(slide, k=0):
    tbls = [sh.table for sh in slide.shapes if sh.has_table]
    return tbls[k]


# --- slide 9: Table 2 (nine models) --------------------------------------
tbl = table_by_idx(S[8])
for i, m in enumerate(DECK_ORDER, start=1):
    vals = [ci(None, m, "train"), ci(None, m, "train", ap=True),
            ci(None, m, "test"), ci(None, m, "test", ap=True),
            ci(None, m, "eICU"), ci(None, m, "eICU", ap=True)]
    for j, v in enumerate(vals, start=1):
        set_cell(tbl.rows[i].cells[j], v)
print("[TABLE] slide 9 Table 2 -> v80")

# --- slide 12: Table 3 (stacking row) ------------------------------------
tbl = table_by_idx(S[11])
vals = [ci(None, "Stacking", "train"), ci(None, "Stacking", "train", ap=True),
        ci(None, "Stacking", "test"), ci(None, "Stacking", "test", ap=True),
        ci(None, "Stacking", "eICU"), ci(None, "Stacking", "eICU", ap=True)]
for j, v in enumerate(vals, start=1):
    set_cell(tbl.rows[1].cells[j], v)
print("[TABLE] slide 12 Table 3 -> v80")

# --- slide 23: headline table --------------------------------------------
tbl = table_by_idx(S[22])
rows = [
    ("Stacking", "MIMIC-IV test", "Stacking", "test"),
    ("Stacking", "eICU-CRD", "Stacking", "eICU"),
    ("MLP", "MIMIC-IV test", "MLP", "test"),
    ("MLP", "eICU-CRD", "MLP", "eICU"),
    ("RF", "MIMIC-IV test", "RF", "test"),
    ("RF", "eICU-CRD", "RF", "eICU"),
    ("XGBoost", "MIMIC-IV test", "XGBoost", "test"),
    ("XGBoost", "eICU-CRD", "XGBoost", "eICU"),
]
for i, (name, coh_lbl, m, coh) in enumerate(rows, start=1):
    r = RES[(RES.Model == m) & (RES.cohort == ("internal test" if coh == "test"
                                               else "eICU external"))].iloc[0]
    ap_v = ci(None, m, coh, ap=True)
    auc_v = ci(None, m, coh)
    if m == "Stacking":
        c = CAL[CAL.cohort == coh].iloc[0]
        slope = f"{c.slope:.2f} ({c.s_lo:.2f}\u2013{c.s_hi:.2f})"
    else:
        slope = f"{r.calib_slope:.2f}"
    vals = [auc_v, ap_v, f"{r.Brier:.3f}", slope]
    for j, v in enumerate(vals, start=2):
        set_cell(tbl.rows[i].cells[j], v)
print("[TABLE] slide 23 headline -> v80")

# --- slide 24: v70 vs v80 selection-strategy comparison ------------------
tbl = table_by_idx(S[23])
hdr = ["Model",
       "Training OOF — before CV", "Training OOF — fold-internal", "Δ",
       "Internal test — before CV", "Internal test — fold-internal", "Δ",
       "eICU-CRD — before CV", "eICU-CRD — fold-internal", "Δ"]
for j, h in enumerate(hdr):
    set_cell(tbl.rows[0].cells[j], h)

def v70_auc(m, coh):
    r = V70M[(V70M.Model == m) & (V70M.cohort == coh)].iloc[0]
    return f"{r.AUC:.3f} ({r.CI_lo:.3f}\u2013{r.CI_hi:.3f})", r.AUC

def v80_auc(m, coh):
    r = AUC[(AUC.Model == m) & (AUC.cohort == coh)].iloc[0]
    return (f"{r.AUC:.3f} ({r.CI_lo:.3f}\u2013{r.CI_hi:.3f})", r.AUC)

models24 = DECK_ORDER + ["Stacking"]
cohorts = [("train", "train"), ("test", "test"), ("eICU", "eICU")]
for i, m in enumerate(models24, start=1):
    set_cell(tbl.rows[i].cells[0], NICE[m])
    j = 1
    for vcoh, _ in cohorts:
        s70, a70 = v70_auc(m, vcoh)
        s80, a80 = v80_auc(m, vcoh)
        d = a80 - a70
        ds = f"{d:+.3f}" if abs(d) >= 0.0005 else "0.000"
        set_cell(tbl.rows[i].cells[j], s70)
        set_cell(tbl.rows[i].cells[j + 1], s80)
        set_cell(tbl.rows[i].cells[j + 2], ds)
        j += 3
set_text(shape_by_id(S[23], 2),
    "Table — Previous analysis (selection before CV) vs revised "
    "(fold-internal re-selection): AUC (95% CI), all ten models")
set_text(shape_by_id(S[23], 4),
    "Same split (seed 42) and identical split-first imputation; the revision "
    "re-runs the univariable screen and the LASSO 1-SE rule inside every "
    "cross-validation fold and evaluates one frozen predictor. Stacking and "
    "top-3 base learners shift by ≤ 0.003 in every cohort (largest shifts in "
    "the weakest learners: decision tree −0.015 internal / −0.013 external); "
    "all reported metrics are now strictly leakage-free.")
print("[TABLE] slide 24 v70-vs-v80 rebuilt")

prs.save(str(DECK))
print("[SAVED]", DECK)
