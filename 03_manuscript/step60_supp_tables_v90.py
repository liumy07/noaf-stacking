# -*- coding: utf-8 -*-
"""step60_supp_tables_v90.py

Rebuild the two submission-package supplementary tables for the v90-primary
manuscript:

  * Supplementary Table S4  — paired DeLong, stacking vs each base learner
  * Supplementary Table S10 — primary (fold-internal imputation) vs
                              sensitivity (imputer fitted once before CV)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.shared import Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR  # noqa: E402  (NOAF_RUN=v90 -> v90_primary)

OUT = V80_DIR
DISPLAY = {"MLP": "Multilayer perceptron", "LightGBM": "LightGBM",
           "XGBoost": "XGBoost", "RF": "Random forest",
           "Logistic": "Logistic regression", "DT": "Decision tree",
           "SVM": "Support vector machine", "Ridge": "Ridge regression",
           "KNN": "K-nearest neighbours", "Stacking": "Stacking ensemble"}
COHLAB = {"train": "Training set, out-of-fold",
          "test": "MIMIC-IV internal test",
          "eICU": "eICU-CRD external",
          "train OOF": "Training set, out-of-fold",
          "internal test": "MIMIC-IV internal test",
          "eICU external": "eICU-CRD external"}
COH_LONG = {"test": "MIMIC-IV internal test (n = 10,316)",
            "eICU": "eICU-CRD external (n = 32,067)"}


def base_doc(title):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Arial"
    st.font.size = Pt(9)
    doc.add_paragraph(title).bold = True
    t = doc.add_table(rows=1, cols=4)
    t.style = "Light Grid Accent 1"
    return doc, t


def main():
    # ---------------- Supplementary Table S4 (DeLong) -------------------
    dl = pd.read_csv(OUT / "v80_delong_paired.csv", encoding="utf-8-sig")
    doc, t = base_doc("Supplementary Table S4. DeLong test for paired comparison of the area under "
                      "the receiver operating characteristic curve: stacking ensemble vs. each base "
                      "learner (primary analysis, imputer refitted inside every cross-validation fold).")
    for j, h in enumerate(["Comparison", "Delta AUC", "95% CI", "P value"]):
        t.rows[0].cells[j].text = h
    for _, r in dl.iterrows():
        cells = t.add_row().cells
        cells[0].text = f"Stacking vs. {DISPLAY[r.base]}, {COH_LONG[r.cohort]}"
        cells[1].text = f"{r.delta:+.4f}"
        cells[2].text = f"{r.lo:+.4f} to {r.hi:+.4f}"
        cells[3].text = "P < 0.001" if r.p < 1e-3 else f"P = {r.p:.3f}"
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run("Note: ").bold = True
    x = dl[(dl.base == "XGBoost") & (dl.cohort == "eICU")].iloc[0]
    p.add_run("Delta AUC = AUC(stacking) \u2212 AUC(base learner); positive values favour the stacking "
              "ensemble. All P values are from the DeLong test for correlated receiver operating "
              f"characteristic curves. The eICU-CRD comparison with XGBoost has Delta AUC {x.delta:+.4f} "
              f"with a 95% CI of {x.lo:+.4f} to {x.hi:+.4f} (P = {x.p:.3f}), i.e. the confidence interval "
              "crosses zero.")
    f4 = OUT / "Supplementary_Table_S4_DeLong_v90.docx"
    doc.save(f4)

    # ---------------- Supplementary Table S10 (sensitivity) -------------
    cmp_ = pd.read_csv(OUT / "v90_sensitivity_AUC.csv", encoding="utf-8-sig")
    doc, t = base_doc("Supplementary Table S10. Sensitivity analysis: freezing the missForest-type "
                      "imputer on the full training set before cross-validation (sensitivity) instead "
                      "of refitting it inside every fold (primary analysis).")
    hdr = ["Model", "Evaluation set", "AUC, primary analysis",
           "AUC, sensitivity analysis", "Difference (sensitivity \u2212 primary)"]
    t2 = doc.add_table(rows=1, cols=5)
    t2.style = "Light Grid Accent 1"
    for j, h in enumerate(hdr):
        t2.rows[0].cells[j].text = h
    for _, r in cmp_.iterrows():
        cells = t2.add_row().cells
        cells[0].text = DISPLAY[r.Model] + (" *" if r.Model == "Stacking" else "")
        cells[1].text = COHLAB[r.cohort]
        cells[2].text = f"{r.AUC_v90:.4f}"     # v90 = primary
        cells[3].text = f"{r.AUC_v80:.4f}"     # v80 = sensitivity
        cells[4].text = f"{-r.delta:+.4f}"     # sensitivity - primary
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.add_run("Note: ").bold = True
    p.add_run("The primary analysis refits the imputer inside every cross-validation fold on that "
              "fold's training rows; the sensitivity analysis fits it once on the 70% training set and "
              "applies it frozen, as in our original pipeline. The frozen predictor evaluated on the "
              "internal test set and eICU-CRD uses a training-only imputer in both versions, so the two "
              "differ only in the out-of-fold predictions that train the meta-learner and rank the base "
              "learners. Across all ten models and three evaluation sets the absolute difference in AUC "
              "did not exceed 0.001. * The paired DeLong test of the two stacking versions gives the "
              "deltas listed for the stacking row.")
    f10 = OUT / "Supplementary_Table_S10_sensitivity_v90.docx"
    doc.save(f10)
    print("[SAVED]", f4.name, "|", f10.name)
    print(t2.rows[11].cells[0].text, "|", " ".join(c.text for c in t2.rows[11].cells[2:]))


if __name__ == "__main__":
    main()
