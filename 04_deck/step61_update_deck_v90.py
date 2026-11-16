# -*- coding: utf-8 -*-
"""step61_update_deck_v90.py

Move the deck to the v90 (fold-internal imputation) primary results:
  * every embedded picture is swapped for the v90 render of the same figure
    (matched by image hash, so inserted slides do not break the mapping);
  * captions, Table 2, Table 3, the headline table, the S4 tables and the
    primary-vs-sensitivity page are rewritten from the v90 CSVs.

Pictures stay native and draggable; no layout is moved.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Emu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR  # noqa: E402  (NOAF_RUN=v90 -> v90_primary)

FIG = V80_DIR / "figures"
OLD_FIG = Path(__file__).resolve().parent / "v80_foldwise" / "figures"
V70 = Path(__file__).resolve().parent / "v70_leakfree"
DECK = Path(r"C:\Users\localuser\Desktop\A prediction model for "
            r"new-onset atrial fibrillation in critically ill patients "
            r"based on stacking ensemble strategy\新结果figure设计_修订版.pptx")
BAK = Path(__file__).resolve().parent / "_backup_deck_pre_v90.pptx"

AUC = pd.read_csv(V80_DIR / "v80_ci_auc.csv")
AP = pd.read_csv(V80_DIR / "v80_ci_ap.csv")
RES = pd.read_csv(V80_DIR / "v80_results.csv")
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv")
DL = pd.read_csv(V80_DIR / "v80_delong_paired.csv")
SENS = pd.read_csv(V80_DIR / "v90_sensitivity_AUC.csv")
V70M = pd.read_csv(V70 / "v70_nine_models_metrics.csv")

DECK_ORDER = ["MLP", "XGBoost", "RF", "LightGBM", "Logistic", "SVM",
              "Ridge", "DT", "KNN"]
NICE = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
        "RF": "Random forest", "LightGBM": "LightGBM",
        "Logistic": "Logistic regression", "SVM": "Support vector machine",
        "Ridge": "Ridge regression", "DT": "Decision tree",
        "KNN": "K-nearest neighbours"}
KEY = {v: k for k, v in NICE.items()}


def ci(m, coh, ap=False):
    df = AP if ap else AUC
    r = df[(df.Model == m) & (df.cohort == coh)].iloc[0]
    lo, hi = (r.lo, r.hi) if ap else (r.CI_lo, r.CI_hi)
    col = "AP" if ap else "AUC"
    return f"{r[col]:.3f} ({lo:.3f}\u2013{hi:.3f})"


def set_text(shape, text):
    tf = shape.text_frame
    p = tf.paragraphs[0]
    for extra in list(tf.paragraphs)[1:]:
        extra._p.getparent().remove(extra._p)
    if p.runs:
        p.runs[0].text = text
        for r in list(p.runs)[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run(text)


def set_cell(cell, text):
    set_text(cell, text)


def replace_pic(slide, pic, img_path):
    _, rId = slide.part.get_or_add_image_part(str(img_path))
    blip = pic._element.blipFill.find(qn("a:blip"))
    blip.set(qn("r:embed"), rId)
    bf = pic._element.blipFill
    src = bf.find(qn("a:srcRect"))
    if src is not None:
        bf.remove(src)
    with Image.open(img_path) as im:
        w, h = im.size
    new_h = int(pic.width * h / w)
    top = pic.top + (pic.height - new_h) // 2
    pic.height = new_h
    pic.top = max(Emu(180000), top)


def shape_by_id(slide, sid):
    for sh in slide.shapes:
        if sh.shape_id == sid:
            return sh
    raise KeyError(sid)


def main():
    if not BAK.exists():
        shutil.copy2(DECK, BAK)
        print("[BACKUP]", BAK.name)
    prs = Presentation(str(DECK))
    S = list(prs.slides)

    # ---------------------------------------------------------- pictures
    src = {}
    for f in OLD_FIG.glob("*.png"):
        src[hashlib.md5(f.read_bytes()).hexdigest()] = f.stem
    swapped = missing = 0
    for si, slide in enumerate(S, start=1):
        for sh in slide.shapes:
            if sh.shape_type != 13:
                continue
            stem = src.get(hashlib.md5(sh.image.blob).hexdigest())
            if stem is None:
                missing += 1
                print(f"  [SKIP] slide {si}: picture not matched to a source figure")
                continue
            target = FIG / f"{stem}.png"
            if not target.exists():
                print(f"  [MISS] {stem} not produced by the v90 run")
                continue
            replace_pic(slide, sh, target)
            swapped += 1
    print(f"[PICS] swapped {swapped}, unmatched {missing}")

    # slide 11 (enlarged stacking workflow) embeds an older render whose hash
    # matches no source file; replace it explicitly with the v90 workflow.
    for sh in S[10].shapes:
        if sh.shape_type == 13:
            replace_pic(S[10], sh, FIG / "Figure_Workflow_stacking.png")
            print("[PIC] slide 11 -> Figure_Workflow_stacking (explicit)")
            break

    # ---------------------------------------------------------- captions
    set_text(shape_by_id(S[0], 4),
             "Study flowchart \u2014 split-first imputation with the imputer refitted inside every "
             "cross-validation fold (no leakage at any stage); univariable screening and LASSO "
             "1-SE re-run inside the same folds (26 variables retained); top-3 base learners "
             "locked by training-set out-of-fold AUC; one frozen predictor used for every "
             "evaluation and SHAP.")

    set_text(shape_by_id(S[12], 4),
             "Calibration with bootstrap 95% CI band and intercept/slope 95% CI \u2014 "
             "internal calibration good (slope 1.14, 95% CI 1.10\u20131.19) though not "
             "perfect; eICU-CRD substantially miscalibrated (slope 0.72, intercept "
             "\u22121.73), recalibration required before absolute-risk use.")

    set_text(shape_by_id(S[15], 4),
             "Figure 6 \u2014 internal hold-out test set. Stacking: AUC 0.843 (95% CI "
             "0.835\u20130.851), AP 0.668 (95% CI 0.651\u20130.684); net benefit at 10% +0.220 "
             "(95% CI +0.218 to +0.222) and at 20% +0.174 (95% CI +0.170 to +0.178). "
             "All AUC/AP values carry 95% CI in the legends; the DCA box carries "
             "bootstrap 95% CI.")

    set_text(shape_by_id(S[16], 75),
             "Figure 7 \u2014 eICU-CRD external validation. Stacking: AUC 0.753 (95% CI "
             "0.743\u20130.762), AP 0.201 (95% CI 0.190\u20130.213); net benefit at 10% +0.004 "
             "(95% CI +0.003 to +0.005) and at 20% \u22120.025 (95% CI \u22120.027 to \u22120.024). "
             "All CIs shown in legends/box.")

    set_text(shape_by_id(S[22], 4),
             "Figure 6F \u2014 sampling stability of the SHAP feature ranking: 10 "
             "independent re-drawn explain sets (same 300-patient protocol); Spearman "
             "\u03c1 vs the main analysis, mean 0.977 (min 0.969); top-10 Jaccard 0.821 \u2014 "
             "the ranking is stable.")

    # ---------------------------------------------------------- tables
    # Table 2 (nine models)
    tbl = [sh.table for sh in S[8].shapes if getattr(sh, "has_table", False) and sh.has_table][0]
    for i, m in enumerate(DECK_ORDER, start=1):
        vals = [ci(m, "train"), ci(m, "train", True), ci(m, "test"),
                ci(m, "test", True), ci(m, "eICU"), ci(m, "eICU", True)]
        for j, v in enumerate(vals, start=1):
            set_cell(tbl.rows[i].cells[j], v)
    print("[TABLE] Table 2")

    # Table 3 (stacking row)
    tbl = [sh.table for sh in S[11].shapes if getattr(sh, "has_table", False) and sh.has_table][0]
    vals = [ci("Stacking", "train"), ci("Stacking", "train", True),
            ci("Stacking", "test"), ci("Stacking", "test", True),
            ci("Stacking", "eICU"), ci("Stacking", "eICU", True)]
    for j, v in enumerate(vals, start=1):
        set_cell(tbl.rows[1].cells[j], v)
    print("[TABLE] Table 3")

    # Supplementary Table S4 (two tables on slide 15)
    tbls = [sh.table for sh in S[14].shapes if getattr(sh, "has_table", False) and sh.has_table]
    for tbl, coh in zip(tbls, ["test", "eICU"]):
        for row in list(tbl.rows)[1:]:
            label = row.cells[0].text.strip()
            base = KEY[label]
            r = DL[(DL.base == base) & (DL.cohort == coh)].iloc[0]
            set_cell(row.cells[1], f"{r.delta:+.4f} ({r.lo:+.4f} to {r.hi:+.4f})")
            set_cell(row.cells[2], "P < 0.001" if r.p < 1e-3 else f"P = {r.p:.3f}")
    x = DL[(DL.base == "XGBoost") & (DL.cohort == "eICU")].iloc[0]
    set_text(shape_by_id(S[14], 8),
             f"The eICU-CRD comparison with XGBoost has \u0394AUC {x.delta:+.4f} with a 95% CI "
             f"of {x.lo:+.4f} to {x.hi:+.4f} (P = {x.p:.3f}), i.e. the confidence interval "
             "crosses zero.")
    print("[TABLE] Supplementary Table S4")

    # Headline table (slide 24)
    tbl = [sh.table for sh in S[23].shapes if getattr(sh, "has_table", False) and sh.has_table][0]
    rows = [("Stacking", "test"), ("Stacking", "eICU"), ("MLP", "test"),
            ("MLP", "eICU"), ("RF", "test"), ("RF", "eICU"),
            ("XGBoost", "test"), ("XGBoost", "eICU")]
    for i, (m, coh) in enumerate(rows, start=1):
        r = RES[(RES.Model == m) & (RES.cohort == ("internal test" if coh == "test"
                                                   else "eICU external"))].iloc[0]
        if m == "Stacking":
            c = CAL[CAL.cohort == coh].iloc[0]
            slope = f"{c.slope:.2f} ({c.s_lo:.2f}\u2013{c.s_hi:.2f})"
        else:
            slope = f"{r.calib_slope:.2f}"
        vals = [ci(m, coh), ci(m, coh, True), f"{r.Brier:.3f}", slope]
        for j, v in enumerate(vals, start=2):
            set_cell(tbl.rows[i].cells[j], v)
    print("[TABLE] headline")

    # Previous-analysis vs revised table (slide 25): refresh revised -> v90
    tbl = [sh.table for sh in S[24].shapes if getattr(sh, "has_table", False) and sh.has_table][0]
    for i, m in enumerate(DECK_ORDER + ["Stacking"], start=1):
        j = 1
        for coh in ["train", "test", "eICU"]:
            r70 = V70M[(V70M.Model == m) & (V70M.cohort == coh)].iloc[0]
            r90 = AUC[(AUC.Model == m) & (AUC.cohort == coh)].iloc[0]
            d = r90.AUC - r70.AUC
            set_cell(tbl.rows[i].cells[j], f"{r70.AUC:.3f} ({r70.CI_lo:.3f}\u2013{r70.CI_hi:.3f})")
            set_cell(tbl.rows[i].cells[j + 1], f"{r90.AUC:.3f} ({r90.CI_lo:.3f}\u2013{r90.CI_hi:.3f})")
            set_cell(tbl.rows[i].cells[j + 2],
                     f"{d:+.3f}" if abs(d) >= 0.0005 else "0.000")
            j += 3
    print("[TABLE] previous-vs-revised")

    # Primary vs sensitivity page (slide 26)
    set_text(shape_by_id(S[25], 2),
             "Sensitivity analysis  \u00b7  imputer fitted once on the full training set "
             "before cross-validation (Supplementary Table S10)")
    set_text(shape_by_id(S[25], 3),
             "Primary analysis: the missForest-type imputer is refitted inside every "
             "cross-validation fold on that fold's training rows. Sensitivity analysis: "
             "the imputer is fitted once on the 70% training set and applied frozen, as in "
             "our original pipeline. The frozen predictor used for the internal test set "
             "and eICU-CRD is identical in construction in both versions.")
    tbl = [sh.table for sh in S[25].shapes if getattr(sh, "has_table", False) and sh.has_table][0]
    hdr = ["Evaluation set", "AUC, primary analysis",
           "AUC, imputer fitted once (sensitivity)", "\u0394AUC (primary \u2212 sensitivity)", "P"]
    for j, h in enumerate(hdr):
        set_cell(tbl.rows[0].cells[j], h)
    for row in list(tbl.rows)[1:]:
        label = row.cells[0].text.strip()
        coh = ("train OOF" if "out-of-fold" in label else
               "internal test" if "internal" in label else "eICU external")
        r = SENS[(SENS.cohort == coh) & (SENS.Model == "Stacking")].iloc[0]
        set_cell(row.cells[1], f"{r.AUC_v90:.4f}")
        set_cell(row.cells[2], f"{r.AUC_v80:.4f}")
    set_text(shape_by_id(S[25], 5),
             "No single model changed by more than 0.001 in any evaluation set (all nine "
             "models: Supplementary Table S10). The primary analysis is therefore the fully "
             "leakage-free pipeline, and the earlier pipeline, in which imputation preceded "
             "cross-validation, is reported here as the sensitivity analysis.")
    print("[TABLE] sensitivity page")

    alt = DECK.with_name(DECK.stem + "_v90.pptx")
    try:
        prs.save(str(DECK))
        print("[SAVED]", DECK.name)
    except PermissionError:
        prs.save(str(alt))
        print("[LOCKED] deck is open elsewhere; wrote", alt.name)
        try:
            shutil.copy2(alt, DECK)
            print("[COPIED] over the original deck")
        except PermissionError:
            print("[STILL LOCKED] close PowerPoint and copy manually")


if __name__ == "__main__":
    main()
