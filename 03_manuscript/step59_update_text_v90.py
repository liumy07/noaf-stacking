# -*- coding: utf-8 -*-
"""step59_update_text_v90.py

Rewrite every Results/Abstract/Methods/Discussion statement that carries a
number affected by the switch to the fold-internal-imputation (v90) pipeline.

Paragraphs without EndNote fields are rebuilt inside their first run (all runs
in those paragraphs share one format).  Paragraphs that do contain EndNote
fields (P5, P21, P78) are edited run-by-run so that no w:fldChar/w:instrText
node is touched.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document

DOCX = Path(r"D:/BJND-4002-11项目/BJND-4002-11/NOAF_stacking_manuscript_EndNote_revised_v90.docx")


def rebuild(p, new_text):
    """Replace the whole paragraph text, keeping the first run's formatting."""
    assert p.runs, "paragraph has no runs"
    p.runs[0].text = new_text
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)


def replace_in_runs(p, old, new):
    hits = 0
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new)
            hits += 1
    assert hits == 1, f"expected 1 run containing {old!r}, found {hits}"


P56 = ("We then developed and benchmarked the nine candidate classification algorithms across both "
       "cohorts. Nine classification models - logistic regression, ridge regression, decision tree, "
       "k-nearest neighbours (KNN), support vector machine (SVM), multilayer perceptron (MLP), random "
       "forest (RF), LightGBM and XGBoost - were developed on the MIMIC-IV training set with "
       "hyperparameters pre-specified and held fixed throughout, and were evaluated on the internal 30% "
       "hold-out test set and on the external eICU-CRD cohort (Figure 3; Supplementary Table S3). Ranked "
       "by training-set OOF AUC, which was generated with fold-internal re-selection of predictors, the "
       "MLP performed best (0.837, 95% CI 0.832-0.843), followed by the RF (0.836, 95% CI 0.830-0.841) "
       "and XGBoost (0.833, 95% CI 0.828-0.839); LightGBM ranked fifth (0.832, 95% CI 0.826-0.837) and KNN "
       "last (0.803, 95% CI 0.797-0.809). On the internal test set (n = 10,316) XGBoost achieved the best "
       "discrimination (AUC 0.838, 95% CI 0.830-0.847), followed by the RF (0.838, 95% CI 0.830-0.846), "
       "LightGBM (0.837, 95% CI 0.829-0.845) and the MLP (0.835, 95% CI 0.826-0.843), with KNN performing "
       "worst (0.801, 95% CI 0.792-0.810; Figure 3A and B).")

P57 = ("In the external eICU-CRD cohort (n = 32,067) discrimination of every model decreased but remained "
       "moderate, XGBoost and LightGBM being highest (AUC 0.751, 95% CI 0.742-0.761 and 0.750, 95% CI "
       "0.740-0.760, respectively), followed by the RF (0.746, 95% CI 0.736-0.755) and the MLP (0.744, 95% "
       "CI 0.735-0.754), whereas KNN again ranked last (0.696, 95% CI 0.686-0.706; Table 2). "
       "Precision-recall analysis yielded a concordant picture: on the internal test set the highest average "
       "precisions were observed for XGBoost (0.665, 95% CI 0.650-0.682), LightGBM (0.662, 95% CI "
       "0.647-0.679) and the RF (0.661, 95% CI 0.645-0.678), against an event prevalence of 0.277; in "
       "eICU-CRD the highest average precision was attained by XGBoost (0.216, 95% CI 0.204-0.229) followed "
       "by LightGBM (0.216, 95% CI 0.204-0.229), against a prevalence of 0.077. Conventional logistic "
       "regression remained competitive throughout (internal test AUC 0.833, 95% CI 0.825-0.841; external "
       "AUC 0.738, 95% CI 0.728-0.747). Because the confidence intervals of the leading models overlapped "
       "substantially, model ranking was interpreted with caution. The corresponding training-set OOF curves "
       "are shown in Supplementary Figure S1.")

P61 = ("We combined the three leading algorithms in a two-layer stacking ensemble and evaluated its "
       "discrimination, precision-recall behaviour, calibration and clinical utility. The MLP, RF and "
       "XGBoost served as level-0 base learners, and a LASSO-penalised logistic regression trained on their "
       "strictly out-of-fold predictions served as the level-1 meta-learner (Figure 4A); all reported "
       "performance derives from the single frozen predictor defined in Section 2.5. The meta-learner "
       "assigned positive weights to all three learners (MLP 2.47, RF 2.21 and XGBoost 0.81; intercept "
       "\u22122.72), so no base learner was redundant. On the internal 30% hold-out test set the stacking "
       "ensemble achieved an AUC of 0.843 (95% CI 0.835-0.851; Figure 4B), which exceeded that of each of its "
       "base learners by a margin that was statistically detectable but modest in absolute terms (DeLong "
       "\u0394AUC vs. MLP +0.0083, 95% CI +0.0063 to +0.0103, P < 0.001; vs. XGBoost +0.0046, 95% CI +0.0020 "
       "to +0.0071, P < 0.001; vs. RF +0.0047, 95% CI +0.0027 to +0.0067, P < 0.001). Its average precision "
       "was the highest of all ten models on the internal test set (0.668 (0.651\u20130.684); Figure 4C).")

P62 = ("In the eICU-CRD external-validation cohort the stacking ensemble achieved an AUC of 0.753 (95% CI "
       "0.743-0.762; Figure 5A) and an average precision of 0.201 (95% CI 0.190-0.213; Figure 5B). Here the "
       "advantage over the strongest base learners was smaller and heterogeneous (DeLong \u0394AUC vs. MLP "
       "+0.0082, 95% CI +0.0057 to +0.0107, P < 0.001; vs. XGBoost +0.0013, 95% CI -0.0017 to +0.0044, "
       "P = 0.400; vs. RF +0.0068, 95% CI +0.0044 to +0.0092, P < 0.001), and for average precision the "
       "stacking ensemble (0.201) was below XGBoost (0.216) and LightGBM (0.216). The complete set of paired "
       "comparisons is given in Supplementary Table S4. The stacking ensemble is therefore best described as "
       "discriminating at least as well as the strongest individual learners rather than as uniformly "
       "superior: it was the best of the ten models in both validation cohorts, but in the external cohort it "
       "performed at the level of, and not meaningfully better than, the best base learners.")

P63 = ("Calibration differed sharply between cohorts. In the internal test set the stacking ensemble was "
       "well calibrated overall (intercept +0.115, 95% CI +0.068 to +0.164; slope 1.142, 95% CI 1.097-1.190; "
       "Brier score 0.140; E:O 1.00; Figure 4E): the calibration slope lay close to but statistically "
       "distinguishable from 1; a slope above 1 indicates that the spread of the predicted probabilities was "
       "slightly too narrow - the predictions were not extreme enough rather than over-confident - so the "
       "deviation from ideal calibration was one of mild under-dispersion, and internal calibration was "
       "clearly better than external. In eICU-CRD, calibration was poor: the intercept was strongly negative "
       "(-1.734, 95% CI -1.759 to -1.713) and the slope was substantially below unity (0.722, 95% CI "
       "0.689-0.752), the pattern of predictions that were too extreme in the tails, with an "
       "expected-to-observed ratio of 3.02 and a Brier score of 0.107 (Figure 5D; all calibration metrics in "
       "Supplementary Table S5). In aggregate the model overestimated the number of NOAF events in the "
       "external cohort approximately threefold while still ranking patients in the correct order; this is a "
       "statement about cohort-level calibration, not about a uniform threefold inflation of every individual "
       "patient's risk.")

P64 = ("Decision curve analysis showed that this distinction matters for threshold-based use. In the internal "
       "test set the net benefit of the stacking ensemble exceeded that of both treat-all and treat-none "
       "strategies across the analysed range of threshold probabilities (net benefit 0.220 at a threshold of "
       "0.10, 0.174 at 0.20, 0.134 at 0.30, 0.104 at 0.40 and 0.074 at 0.50; Figure 4D). In eICU-CRD the net "
       "benefit was positive only at threshold probabilities close to the local prevalence (0.004 at a "
       "threshold of 0.10) and became negative above that point (\u22120.025 at 0.20, \u22120.046 at 0.30, "
       "\u22120.060 at 0.40 and \u22120.069 at 0.50; Figure 5C; Supplementary Table S6), although it remained "
       "above the treat-all strategy at every threshold examined. These results indicate that the frozen model "
       "supports threshold-based decision-making in a population resembling the development cohort, but that "
       "its absolute predictions require recalibration, and its decision threshold re-derivation, before it is "
       "used in a cohort with a case mix resembling eICU-CRD. The in-sample training AUC of the stacking "
       "ensemble (0.842 with strictly out-of-fold predictions; 0.944 for the frozen pipeline refitted on the "
       "training set) is reported only to document that overfitting was limited, and no claim of improved "
       "generalisation rests on the training-to-test difference.")

P83 = ("This study has several strengths. It combines heterogeneous base learners in a stacking ensemble "
       "validated across two independent, large critical-care databases; it relies on a parsimonious "
       "26-variable panel composed entirely of variables routinely recorded within the first 24 hours, "
       "favouring feasibility at the bedside; its development protocol was free of leakage at every stage: the "
       "imputer was refitted inside every cross-validation fold on that fold's training rows, the univariable "
       "screen and the LASSO were re-run inside the same folds, base learners were ranked by training-set "
       "out-of-fold performance and locked before the internal test set was opened, and one frozen predictor "
       "generated all reported performance estimates and all interpretations; and its evaluation included "
       "calibration, decision curve analysis with explicit loss-ratio interpretation, and interpretation at "
       "both population and individual-patient levels, including discordant cases.")


def main():
    doc = Document(DOCX)

    rebuild(doc.paragraphs[56], P56)
    rebuild(doc.paragraphs[57], P57)
    rebuild(doc.paragraphs[61], P61)
    rebuild(doc.paragraphs[62], P62)
    rebuild(doc.paragraphs[63], P63)
    rebuild(doc.paragraphs[64], P64)
    rebuild(doc.paragraphs[83], P83)

    # ---- abstract (contains no fields, but keep both runs intact) --------
    p5 = doc.paragraphs[5]
    replace_in_runs(p5, "0.753 (95% CI 0.744-0.763)", "0.753 (95% CI 0.743-0.762)")
    replace_in_runs(p5, "average precisions of 0.668 and 0.202", "average precisions of 0.668 and 0.201")
    replace_in_runs(p5, "internal slope 1.14, 95% CI 1.09-1.19", "internal slope 1.14, 95% CI 1.10-1.19")

    # ---- discussion (has two EndNote fields, edit only the text run) -----
    replace_in_runs(doc.paragraphs[78], "0.668 to 0.202", "0.668 to 0.201")

    # ---- Methods 2.3: flip primary/sensitivity framing -------------------
    p21 = doc.paragraphs[21]
    r6, r7 = p21.runs[6], p21.runs[7]
    assert r6.text.strip().endswith("applied to each cohort in full"), r6.text[-60:]
    r6.text = ("); in contrast to our original analysis, in which imputation was applied to each cohort in "
               "full before any split, the imputation stage was moved inside the analysis pipeline: the "
               "MIMIC-IV cohort was first split 70:30, and the imputer used for the two evaluation cohorts was "
               "fitted on the training rows only and then applied unchanged to the training set, the internal "
               "test set and eICU-CRD, none of which contributed to its estimation. Within the development "
               "loop, the imputer was refitted inside every cross-validation fold on that fold's training "
               "rows, and the univariable screen, the LASSO, the nine learners and the meta-learner were all "
               "estimated inside the same fold, so that every out-of-fold prediction used to rank the "
               "algorithms and to train the meta-learner came from a pipeline that had never seen the held-out "
               "rows at any stage, preprocessing included. As a sensitivity analysis, the imputer was instead "
               "fitted once on the full training set before cross-validation and the whole development loop "
               "repeated; discrimination was essentially unchanged (cross-validated AUC of the stacking "
               "ensemble 0.8416 vs. 0.8416, internal test set 0.8429 vs. 0.8430, eICU-CRD 0.7526 vs. 0.7531, "
               "and no single model changed by more than 0.001 in any evaluation set; Supplementary Table S10), "
               "so the pipeline reported here is the fully leakage-free one.")
    tail = r7.text
    marker = "Because a single completed dataset"
    assert marker in tail
    r7.text = " " + tail[tail.index(marker):]

    bak = DOCX.with_name(DOCX.stem + "_pre_text_v90.docx")
    if not bak.exists():
        shutil.copy2(DOCX, bak)
    doc.save(DOCX)
    print("[SAVED]", DOCX.name)


if __name__ == "__main__":
    main()
