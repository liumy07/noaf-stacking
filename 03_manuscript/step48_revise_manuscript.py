# -*- coding: utf-8 -*-
"""step48_revise_manuscript.py — revise the NOAF manuscript to the v80
(fold-internal selection + frozen predictor) results and the second-round
reviewer issues.

Rules:
  * EndNote citation fields are NEVER touched. Paragraphs containing fields
    are rewritten by segmented replacement around the citation blocks.
  * Plain paragraphs are replaced whole (first-run formatting preserved).
  * Numbers elsewhere are patched by surgical, formatting-preserving
    substring replacement inside non-field runs only.
  * Discussion paragraphs 84-87 (duplicated old text, MA05) are deleted.
  * Table 2 is rebuilt from the v80 CI tables. Figures 1-6 are replaced /
    inserted with the v80 figure set (MO16 layout).
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR  # noqa: E402
from docx import Document  # noqa: E402
from docx.oxml.ns import qn  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Emu  # noqa: E402

DOC = Path(r"D:\BJND-4002-11项目\BJND-4002-11\NOAF_stacking_manuscript_EndNote_revised.docx")
BAK = Path(r"C:\Users\localuser\WorkBuddy\2026-09-09-17-02-31\output\rev_noaf\_backup_manuscript_pre_v80.docx")
shutil.copy2(DOC, BAK)
print("backup ->", BAK)

FIG = V80_DIR / "figures"
D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
AUC = pd.read_csv(V80_DIR / "v80_ci_auc.csv", encoding="utf-8-sig").set_index(["Model", "cohort"])
APC = pd.read_csv(V80_DIR / "v80_ci_ap.csv", encoding="utf-8-sig").set_index(["Model", "cohort"])
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv", encoding="utf-8-sig").set_index("cohort")
DL = pd.read_csv(V80_DIR / "v80_delong_paired.csv", encoding="utf-8-sig")
RANK = pd.read_csv(V80_DIR / "v80_shap_ranking.csv", encoding="utf-8-sig")
CASES = pd.read_csv(V80_DIR / "v80_shap_cases.csv", encoding="utf-8-sig").set_index("CaseType")
STAB = pd.read_csv(V80_DIR / "v80_shap_stability_summary.csv", encoding="utf-8-sig")
SHZ = np.load(V80_DIR / "v80_shap_values.npz", allow_pickle=True)
from scipy.stats import spearmanr  # noqa: E402
_rho_pl = spearmanr(pd.Series(np.abs(SHZ["sv_prob"]).mean(0)).rank(),
                    pd.Series(np.abs(SHZ["sv_logit"]).mean(0)).rank()).statistic
BASE_VAL = float(SHZ["base_val_prob"])


def fmt(m, coh, col="AUC", ci=None):
    if ci is None:
        ci = ("CI_lo", "CI_hi") if col == "AUC" else ("lo", "hi")
    r = (AUC if col == "AUC" else APC).loc[(m, coh)]
    return f"{r[col if col=='AUC' else 'AP']:.3f} ({r[ci[0]]:.3f}\u2013{r[ci[1]]:.3f})"


def dl(m, coh):
    r = DL[(DL.cohort == coh) & (DL.base == m)].iloc[0]
    p = "P < 0.001" if r.p < 1e-3 else f"P = {r.p:.3f}"
    s = "+" if r.delta >= 0 else "\u2212"
    return (f"{s}{abs(r.delta):.4f}, 95% CI {s}{abs(r.lo):.4f} to "
            f"{s}{abs(r.hi):.4f}, {p}")


# ---------------------------------------------------------------- helpers
def runs_with_field_state(p):
    out, depth, in_result = [], 0, False
    for r in p.runs:
        flds = r._element.findall(qn("w:fldChar"))
        is_field = bool(flds) or depth > 0
        for f in flds:
            t = f.get(qn("w:fldCharType"))
            if t == "begin":
                depth += 1
            elif t == "separate":
                in_result = True
            elif t == "end":
                depth -= 1
                if depth == 0:
                    in_result = False
        out.append((r, is_field))
    return out


def rewrite_plain(p, text):
    """Full replacement of a paragraph WITHOUT fields; keeps first-run format."""
    assert not any(f for _, f in runs_with_field_state(p)), "has fields"
    runs = list(p.runs)
    for r in runs[1:]:
        r._element.getparent().remove(r._element)
    runs[0].text = text
    assert p.text == text


def rewrite_with_fields(p, text, tokens):
    """Segmented replacement around citation field blocks.

    tokens: rendered citation strings in order of appearance, e.g. ["[11]"].
    """
    chunks = []
    rest = text
    for tok in tokens:
        pre, _, rest = rest.partition(tok)
        assert pre is not None and tok in text, f"token {tok} missing"
        chunks.append(pre)
    chunks.append(rest)
    blocks, cur = [], []
    state = runs_with_field_state(p)
    for r, is_field in state:
        if is_field:
            if cur:
                blocks.append(("text", cur))
                cur = []
            if blocks and blocks[-1][0] == "field":
                blocks[-1][1].append(r)   # merge runs of the SAME field block
            else:
                blocks.append(("field", [r]))
        else:
            cur.append(r)
    if cur:
        blocks.append(("text", cur))
    n_fields = sum(1 for k, _ in blocks if k == "field")
    assert n_fields == len(tokens), f"{n_fields} fields vs {len(tokens)} tokens"
    ti = 0
    for kind, rs in blocks:
        if kind == "field":
            continue
        chunk = chunks[ti] if ti < len(chunks) else ""
        ti += 1
        if rs:
            rs[0].text = chunk
            for r in rs[1:]:
                r.text = ""
    assert p.text == text, f"mismatch:\n{p.text[:200]}\n{text[:200]}"


def surgical(p, old, new):
    """Formatting-preserving substring replacement in non-field runs."""
    spans, pos = [], 0
    for r, is_field in runs_with_field_state(p):
        if is_field:
            continue
        spans.append((r, pos, pos + len(r.text)))
        pos += len(r.text)
    full = "".join(r.text for r, a, b in spans)
    idx = full.find(old)
    assert idx >= 0, f"anchor not found: {old[:60]!r}"
    end = idx + len(old)
    first = True
    for r, a, b in spans:
        if b <= idx or a >= end:
            continue
        pre = r.text[:max(0, idx - a)]
        post = r.text[max(0, end - a):] if end <= b else ""
        if first:
            r.text = pre + new + post
            first = False
        else:
            r.text = pre + post
    assert old not in p.text or old in new


# ---------------------------------------------------------------- load doc
doc = Document(str(DOC))
P = list(doc.paragraphs)

# =========================================================== 1. ABSTRACT
rewrite_plain(P[4],
    "Methods: We conducted a retrospective cohort study using two large, "
    "public critical-care databases: MIMIC-IV (34,384 ICU admissions) and the "
    "eICU Collaborative Research Database (eICU-CRD; 32,067 ICU patients). "
    "Thirty-nine harmonised variables recorded within the first 24 hours of "
    "ICU admission were screened by univariable logistic regression and "
    "reduced by least absolute shrinkage and selection operator (LASSO) "
    "regression under the one-standard-error rule. The MIMIC-IV cohort was "
    "split 70:30 before any imputation; a random-forest-based "
    "chained-equation imputer was fitted on the training set only and applied "
    "unchanged to the internal test set and to eICU-CRD. Nine classification "
    "algorithms were developed on the training set (n = 24,068; internal test "
    "n = 10,316), with the univariable screen and the LASSO re-run inside "
    "every cross-validation fold so that the out-of-fold predictions carried "
    "no selection leakage; the three best-performing algorithms, ranked by "
    "training-set out-of-fold AUC, were combined with a LASSO-penalised "
    "meta-learner into a single frozen predictor used for every evaluation "
    "and every interpretation. The eICU-CRD cohort was reserved in its "
    "entirety for external validation and took no part in any "
    "model-development decision. Discrimination, precision-recall behaviour, "
    "calibration and clinical utility were assessed, and SHapley Additive "
    "exPlanations (SHAP) were used for global and individual-patient "
    "interpretation.")

rewrite_plain(P[5],
    "Results: NOAF occurred in 9,535 patients (27.7%) in MIMIC-IV and 2,457 "
    "patients (7.7%) in eICU-CRD. Of the 39 candidate variables, 37 were "
    "significant on univariable analysis and 26 were retained by LASSO. The "
    "stacking ensemble achieved an area under the receiver operating "
    "characteristic curve (AUC) of 0.843 (95% CI 0.835-0.851) on the internal "
    "test set and 0.753 (95% CI 0.744-0.763) in the external eICU-CRD "
    "cohort, with average precisions of 0.668 and 0.202, respectively. Its "
    "discrimination exceeded that of each of its three base learners on the "
    "internal test set (DeLong P < 0.001); in the external cohort the "
    "advantage was smaller and was statistically detectable against the "
    "multilayer perceptron and the random forest but not against XGBoost. "
    "Calibration was clearly better internally than externally (internal "
    f"slope {CAL.loc['test','slope']:.2f}, 95% CI {CAL.loc['test','s_lo']:.2f}-"
    f"{CAL.loc['test','s_hi']:.2f}; external slope "
    f"{CAL.loc['eICU','slope']:.2f}, 95% CI {CAL.loc['eICU','s_lo']:.2f}-"
    f"{CAL.loc['eICU','s_hi']:.2f}) and the expected-to-observed event ratio "
    f"was {CAL.loc['eICU','E_O']:.2f} in eICU-CRD, indicating a roughly "
    "threefold cohort-level overestimation of absolute risk. Decision curve "
    "analysis showed a positive net benefit across threshold probabilities "
    "of 0.10-0.50 internally, but no net benefit above a threshold of about "
    "0.10 in eICU-CRD. SHAP analysis identified age, maximum heart rate, "
    "congestive heart failure and vasopressor exposure as the dominant "
    "predictors, and this ranking was reproduced across repeated subsamples.")

# =========================================================== 2. METHODS
rewrite_with_fields(P[21],
    "Variables with a missing rate exceeding 30% were excluded before "
    "imputation; eight laboratory variables exceeded this threshold (oxygen "
    "saturation, creatine kinase-MB, N-terminal pro-B-type natriuretic "
    "peptide and creatine phosphokinase, each as maximum and minimum) and "
    "were removed. Before imputation, 26 of the 39 retained predictors had "
    "missing values in MIMIC-IV (median 1.2% among incomplete variables, "
    "maximum 6.5% for total calcium) and 27 in eICU-CRD (median 2.7%, "
    "maximum 21.7% for the maximum anion gap); the complete variable-wise "
    "missingness pattern is given in Supplementary Table S8. All remaining "
    "missing values were completed by random-forest-based chained-equation "
    "single imputation (a missForest-type procedure [11]); in contrast to "
    "our original analysis, in which imputation was applied to each cohort "
    "in full before any split, the imputation stage was moved inside the "
    "analysis pipeline: the MIMIC-IV cohort was first split 70:30, the "
    "imputer was fitted on the training rows only, and the frozen imputer "
    "was then applied unchanged to the training set, the internal test set "
    "and eICU-CRD, none of which contributed to its estimation. No "
    "imputation was refitted inside the cross-validation folds. Because a "
    "single completed dataset was produced per split rather than multiple "
    "imputations, the confidence intervals reported below do not reflect "
    "imputation uncertainty; given the low missingness of the retained "
    "predictors this is unlikely to alter the conclusions (Discussion).",
    ["[11]"])

rewrite_plain(P[24],
    "The MIMIC-IV cohort was randomly split, in a 70:30 ratio with "
    "stratification on the outcome, into a training set (n = 24,068) and an "
    "internal hold-out test set (n = 10,316) before any imputation or other "
    "preprocessing; the eICU-CRD cohort was reserved in its entirety for "
    "external validation. Variable screening proceeded in two steps, both "
    "performed on the training set only. First, each of the 39 candidate "
    "variables was assessed in a univariable logistic regression model, and "
    "variables with P < 0.05 were retained for further selection. This step "
    "is a pragmatic dimension-reduction device rather than a principled "
    "selection criterion: it may discard variables with weak marginal but "
    "strong conditional effects, and the retained set depends on sample "
    "size and event rate. Because the screening was confined to the "
    "training data it cannot inflate the reported test performance, and its "
    "consequences for the final panel are considered in the Discussion.")

rewrite_with_fields(P[25],
    "Second, the retained variables were entered into a least absolute "
    "shrinkage and selection operator (LASSO) logistic regression [13] "
    "implemented in scikit-learn (LogisticRegressionCV), with 10-fold "
    "cross-validation over a grid of 60 logarithmically spaced values of "
    "the inverse regularisation strength C (10\u207b\u2074 to 10\u00b9) and "
    "the area under the receiver operating characteristic curve (AUC) as "
    "the optimisation criterion. The penalty was chosen by the "
    "one-standard-error (1-SE) rule [14], which selects the most "
    "regularised model whose cross-validated performance lies within one "
    "standard error of the optimum, thereby favouring parsimony and "
    "stability over marginally higher cross-validated performance. To give "
    "model ranking an unbiased basis, this entire two-step screening "
    "procedure - the univariable screen and the LASSO with its 1-SE rule - "
    "was additionally re-run from scratch inside every cross-validation "
    "training fold when the out-of-fold predictions of the candidate models "
    "were generated (Section 2.5), so that each held-out fold was scored by "
    "models whose predictor panel had been derived without any reference to "
    "that fold.",
    ["[13]", "[14]"])

rewrite_plain(P[29],
    "Predictions used for different purposes were generated by distinct, "
    "pre-specified procedures. Out-of-fold (OOF) predictions were used to "
    "rank the base learners and to train the meta-learner: in each of ten "
    "stratified folds the univariable screen and the LASSO 1-SE rule were "
    "re-run on the fold-training patients only, the nine algorithms were "
    "fitted on the fold-selected variables, and the held-out fold was "
    "scored, so that every training patient received predictions from "
    "models that had seen neither that patient nor the selection "
    "information of its fold. After cross-validation, the locked two-step "
    "selection was repeated on the complete training set (yielding the "
    "final panel of 26 variables), each of the nine algorithms was refitted "
    "on the full training set with this panel, and the three best-ranked "
    "learners together with the LASSO meta-learner were frozen as one "
    "deployable predictor. All performance figures reported for the "
    "internal test set and for the external cohort were produced by this "
    "single frozen predictor, and the identical fitted predictor - not a "
    "re-estimation - was used for every SHAP analysis, so that the model "
    "that is interpreted is exactly the model that is evaluated. The random "
    "seed was fixed at 42 throughout, and neither the internal test set nor "
    "the external cohort contributed to any tuning or selection decision.")

rewrite_with_fields(P[31],
    "Because no single algorithm was expected to dominate across cohorts, "
    "the three best-performing algorithms were combined in a two-layer "
    "stacking ensemble [17, 18], and the external validation cohort took no "
    "part in this or in any other model-development decision. Base learners "
    "were ranked by their strictly out-of-fold training-set AUC alone - "
    "each fold scored with predictors re-selected inside that fold - and "
    "the three leaders - the MLP (OOF AUC 0.837), the RF (0.836) and "
    "XGBoost (0.834) - were locked before the internal test set was "
    "examined; OOF AUCs are reported to four decimal places in "
    "Supplementary Table S3, and no two algorithms tied at the third "
    "decimal. LightGBM ranked fifth (OOF AUC 0.831) and was not included. "
    "The three locked base learners served as level-0 models, and their OOF "
    "predictions for the training patients constituted the level-1 feature "
    "space. A LASSO-penalised logistic regression was then trained on these "
    "features as the meta-learner, with its penalty selected by 10-fold "
    "cross-validation under AUC optimisation over a grid of 25 log-spaced "
    "values; because the meta-learner is itself sparse, it can "
    "automatically discard base learners whose predictions add no "
    "information beyond those already retained. The fitted meta-learner "
    "assigned positive weights to all three base learners (MLP 2.36, RF "
    "2.17 and XGBoost 0.96; intercept \u22122.73), indicating that each "
    "contributed complementary information, and the predicted probability "
    "of the ensemble was obtained by passing the base-learner probabilities "
    "to the trained meta-learner. The base learners and the meta-learner "
    "were then frozen as a single predictor; no learner was retained with a "
    "zero weight.",
    ["[17, 18]"])

rewrite_with_fields(P[33],
    "Model discrimination was quantified by the AUC and, in view of the "
    "class imbalance of NOAF, by the average precision (AP), that is, the "
    "area under the precision-recall curve; AP is reported alongside the "
    "event prevalence, which is the value expected from a non-informative "
    "classifier. Ninety-five percent confidence intervals for AUC were "
    "derived with the DeLong method, and those for AP, net benefit and the "
    "calibration metrics from 1,000 stratified bootstrap resamples (seed "
    "42) [19]. Differences in AUC between the stacking ensemble and its "
    "base learners were tested with the DeLong method for paired receiver "
    "operating characteristic curves, complemented by 1,000 paired "
    "stratified bootstrap resamples for the confidence interval of each "
    "difference.",
    ["[19]"])

rewrite_with_fields(P[34],
    "Calibration was assessed for every model in both validation cohorts. A "
    "logistic recalibration model was fitted with the model's linear "
    "predictor as its only covariate, logit(P) = \u03b1 + "
    "\u03b2\u00b7logit(p\u0302), with the calibration intercept \u03b1 and "
    "the calibration slope \u03b2 estimated jointly from the same model "
    "(\u03b1 = 0 and \u03b2 = 1 for a perfectly calibrated model); the "
    "Brier score and the ratio of expected to observed event rates (E:O; 1 "
    "for a perfectly calibrated model) were also computed. Clinical utility "
    "was examined by decision curve analysis, which expresses net benefit "
    "as a function of the threshold probability p\u209c at which a patient "
    "would be considered to warrant intensified rhythm surveillance, and "
    "compares it with the strategies of treating all patients and treating "
    "none [20]. The threshold encodes the relative harm of a false alert: "
    "at threshold p\u209c a false positive is weighted p\u209c/(1 \u2212 "
    "p\u209c), so that a threshold of 0.20, for example, weights one false "
    "alert one quarter as heavily as a missed case, whereas a threshold of "
    "0.50 weights them equally. Net benefit was reported over the analysed "
    "threshold range of 0.10-0.50, and no claim of clinical relevance is "
    "attached to any particular threshold. Because the two cohorts differ "
    "almost fourfold in event rate, thresholds were interpreted relative to "
    "the local prevalence rather than in absolute terms.",
    ["[20]"])

# versions [35] — surgical
surgical(P[35], "Python 3.13 with scikit-learn 1.9, XGBoost 3.4, LightGBM "
                "4.7, SHAP 0.52, statsmodels 0.15, NumPy 2.5, pandas 3.0 and "
                "matplotlib 3.11",
         "Python 3.13.12 with scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM "
         "4.7.0, SHAP 0.52.0, statsmodels 0.15.0, SciPy 1.18.1, NumPy 2.5.2, "
         "pandas 3.0.6 and matplotlib 3.11.1")

# SHAP methods [36] — dynamic base value / protocol
rewrite_with_fields(P[36],
    "Model interpretability was addressed with SHapley Additive "
    "exPlanations (SHAP), a game-theoretic framework that decomposes an "
    "individual prediction into additive feature contributions [21]. SHAP "
    "values were computed with a model-agnostic permutation explainer "
    "applied to the complete frozen stacking predictor - preprocessing, the "
    "three refitted base learners and the LASSO meta-learner - so that the "
    "explained function is bit-identical to the function that produced "
    "every performance estimate. The background distribution used to "
    f"estimate the expected model output comprised 100 randomly sampled "
    f"training-set patients (seed 42), which gave a base value (mean "
    f"predicted probability) of {BASE_VAL:.3f}. Attributions are reported "
    "on the probability scale and are additive in the predicted probability "
    "of NOAF, so that the sum of the feature contributions equals the "
    "difference between an individual prediction and the base value; "
    "additivity was verified numerically to machine precision, and the "
    f"feature ranking obtained on the probability scale agreed with that "
    f"obtained on the log-odds scale (Spearman \u03c1 = {_rho_pl:.2f}). SHAP "
    "values were computed for 300 randomly sampled training-set patients, "
    "and global importance was summarised as a beeswarm plot. The "
    "stability of the global ranking was assessed by repeating the "
    "procedure on ten independent random subsamples of the training set "
    "(100 patients each) and comparing the resulting feature rankings.",
    ["[21]"])

rewrite_plain(P[37],
    "Individual-patient explanations were generated for four pre-specified "
    "case types: a true positive (the correctly classified patient with "
    "NOAF and the highest predicted probability), a true negative (the "
    "correctly classified patient without NOAF and the lowest predicted "
    "probability), a false positive (the patient without NOAF and the "
    "highest predicted probability) and a false negative (the patient with "
    "NOAF and the lowest predicted probability), with case and control "
    "status defined at the conventional threshold of 0.5. Because these "
    "rules select the most confidently classified individuals within each "
    "category, the four waterfalls should be read as extreme high-"
    "confidence illustrations of how the frozen predictor combines features "
    "in individual patients rather than as typical clinical cases; they are "
    "presented together with, not instead of, the population-level summary. "
    "Each case is shown as a waterfall plot of its additive contributions.")

# eICU AF flag + time window (MO01) [20]
rewrite_plain(P[20],
    "The primary outcome was new-onset atrial fibrillation (NOAF), defined "
    "as atrial fibrillation documented after the first 24 hours of ICU "
    "admission in a patient without pre-existing atrial fibrillation; the "
    "first-24-hour baseline window with a deferred outcome window follows "
    "the convention of previous NOAF prediction studies in MIMIC-IV and "
    "eICU-CRD [5, 6]. In MIMIC-IV, atrial fibrillation was identified from "
    "the structured heart-rhythm record of the ICU stay by matching the "
    "string \u201cAF (Atrial Fibrillation)\u201d, and patients carrying a "
    "coded history of atrial fibrillation who were not in atrial "
    "fibrillation during the index record were excluded before outcome "
    "ascertainment. In eICU-CRD, NOAF was ascertained from the coded "
    "atrial-fibrillation flag that had been generated during cohort "
    "extraction from the eICU-CRD diagnosis and charted-rhythm fields, and "
    "every patient with a coded history of atrial arrhythmia was excluded "
    "before outcome ascertainment. The exact fields, string-matching rules "
    "and exclusion logic applied in each database are documented in the "
    "supplementary extraction code and in Supplementary Table S7. Because "
    "both databases record rhythm and diagnoses as coded or free-text "
    "entries rather than as continuous rhythm monitoring, ascertainment may "
    "miss brief paroxysmal episodes and could misclassify pre-existing "
    "arrhythmia; this limitation is considered in the Discussion.")

print("Methods done")

# =========================================================== 3. RESULTS
rewrite_plain(P[51],
    "The 37 significant variables were entered into LASSO logistic "
    "regression with 10-fold cross-validation. The minimum-error solution "
    "was obtained at C = 0.536 (cross-validated AUC 0.833). The 1-SE rule "
    "selected the more strongly regularised solution C = 0.0108 "
    "(equivalently \u03bb = 92.5), at which 26 variables retained non-zero "
    "coefficients (Figure 2B and C); this specification sacrificed only "
    "0.002 of cross-validated AUC relative to the minimum-error model "
    "(0.831 vs. 0.833) while removing nearly one third of the candidate "
    "predictors. The same two-step procedure, re-run inside every "
    "cross-validation fold of the model-development stage, retained between "
    "23 and 26 variables per fold (median 25), indicating that the locked "
    "panel was stable across resampling while the out-of-fold estimates "
    "remained free of selection leakage.")

rewrite_plain(P[52],
    "Among the 26 retained predictors, age (+1.09), maximum heart rate "
    "(+0.58) and congestive heart failure (+0.42) carried the largest "
    "positive standardised coefficients, followed by vasopressor exposure "
    "(+0.27) and weight (+0.23); minimum heart rate (\u22120.11), minimum "
    "sodium (\u22120.08), severe liver disease (\u22120.07), first-day "
    "urine output (\u22120.06) and minimum glucose (\u22120.06) carried "
    "negative coefficients. Because the comorbidity and exposure variables "
    "entered the model as single 0/1 indicators, the sign of each "
    "coefficient is directly interpretable: vasopressor exposure shifts the "
    "predicted log-odds upwards relative to non-exposure, consistent with "
    "the univariable association (vasopressor use OR 2.43). Smaller "
    "positive weights were contributed by minimum blood urea nitrogen "
    "(+0.09), maximum potassium (+0.07), maximum haemoglobin (+0.07) and "
    "invasive mechanical ventilation (+0.06). The final panel spanned "
    "demographics, chronic comorbidities, therapeutic interventions and "
    "routinely recorded vital-sign and laboratory parameters, and was "
    "locked before the internal test set was examined and carried forward "
    "to machine-learning modelling.")

rewrite_plain(P[56],
    "We then developed and benchmarked the nine candidate classification "
    "algorithms across both cohorts. Nine classification models - logistic "
    "regression, ridge regression, decision tree, k-nearest neighbours "
    "(KNN), support vector machine (SVM), multilayer perceptron (MLP), "
    "random forest (RF), LightGBM and XGBoost - were developed on the "
    "MIMIC-IV training set with hyperparameters pre-specified and held "
    "fixed throughout, and were evaluated on the internal 30% hold-out "
    "test set and on the external eICU-CRD cohort (Figure 3; Supplementary "
    "Table S3). Ranked by training-set OOF AUC, which was generated with "
    "fold-internal re-selection of predictors, the MLP performed best "
    "(0.837, 95% CI 0.832-0.842), followed by the RF (0.836, 95% CI "
    "0.831-0.841) and XGBoost (0.834, 95% CI 0.829-0.839); LightGBM ranked "
    "fifth (0.831, 95% CI 0.826-0.837) and KNN last (0.803, 95% CI "
    "0.797-0.808). On the internal test set (n = 10,316) the RF achieved "
    "the best discrimination (AUC 0.838, 95% CI 0.830-0.846), followed by "
    "XGBoost (0.838, 95% CI 0.830-0.846), LightGBM (0.837, 95% CI "
    "0.829-0.845) and the MLP (0.835, 95% CI 0.826-0.843), with KNN "
    "performing worst (0.801, 95% CI 0.792-0.810; Figure 3A and B).")

rewrite_plain(P[57],
    "In the external eICU-CRD cohort (n = 32,067) discrimination of every "
    "model decreased but remained moderate, XGBoost and LightGBM being "
    "highest (AUC 0.752, 95% CI 0.742-0.761 and 0.750, 95% CI 0.740-0.760, "
    "respectively), followed by the RF (0.746, 95% CI 0.736-0.755) and the "
    "MLP (0.744, 95% CI 0.735-0.754), whereas KNN again ranked last (0.696, "
    "95% CI 0.686-0.706; Table 2). Precision-recall analysis yielded a "
    "concordant picture: on the internal test set the highest average "
    "precisions were observed for XGBoost (0.664, 95% CI 0.649-0.680), "
    "LightGBM (0.662, 95% CI 0.647-0.679) and the RF (0.661, 95% CI "
    "0.645-0.678), against an event prevalence of 0.277; in eICU-CRD the "
    "highest average precision was attained by LightGBM (0.216, 95% CI "
    "0.204-0.229) followed by XGBoost (0.214, 95% CI 0.202-0.227), against "
    "a prevalence of 0.077. Conventional logistic regression remained "
    "competitive throughout (internal test AUC 0.833, 95% CI 0.825-0.841; "
    "external AUC 0.738, 95% CI 0.728-0.747). Because the confidence "
    "intervals of the leading models overlapped substantially, model "
    "ranking was interpreted with caution. The corresponding training-set "
    "OOF curves are shown in Supplementary Figure S1.")

rewrite_plain(P[61],
    "We combined the three leading algorithms in a two-layer stacking "
    "ensemble and evaluated its discrimination, precision-recall behaviour, "
    "calibration and clinical utility. The MLP, RF and XGBoost served as "
    "level-0 base learners, and a LASSO-penalised logistic regression "
    "trained on their strictly out-of-fold predictions served as the "
    "level-1 meta-learner (Figure 4A); all reported performance derives "
    "from the single frozen predictor defined in Section 2.5. The "
    "meta-learner assigned positive weights to all three learners (MLP "
    "2.36, RF 2.17 and XGBoost 0.96; intercept \u22122.73), so no base "
    "learner was redundant. On the internal 30% hold-out test set the "
    "stacking ensemble achieved an AUC of 0.843 (95% CI 0.835-0.851; "
    "Figure 4B), which exceeded that of each of its base learners by a "
    "margin that was statistically detectable but modest in absolute terms "
    f"(DeLong \u0394AUC vs. MLP {dl('MLP','test')}; vs. XGBoost "
    f"{dl('XGBoost','test')}; vs. RF {dl('RF','test')}). Its average "
    "precision was the highest of all ten models on the internal test set "
    f"({fmt('Stacking','test',col='AP')}; Figure 4C).")

rewrite_plain(P[62],
    "In the eICU-CRD external-validation cohort the stacking ensemble "
    "achieved an AUC of 0.753 (95% CI 0.744-0.763; Figure 5A) and an "
    f"average precision of {fmt('Stacking','eICU',col='AP')[:-1].split('(')[0].strip()} "
    f"(95% CI {APC.loc[('Stacking','eICU'),'lo']:.3f}-"
    f"{APC.loc[('Stacking','eICU'),'hi']:.3f}; Figure 5B). Here the "
    "advantage over the strongest base learners was smaller and "
    f"heterogeneous (DeLong \u0394AUC vs. MLP {dl('MLP','eICU')}; vs. "
    f"XGBoost {dl('XGBoost','eICU')}; vs. RF {dl('RF','eICU')}), and for "
    "average precision the stacking ensemble (0.202) was below XGBoost "
    "(0.214) and LightGBM (0.216). The complete set of paired comparisons "
    "is given in Supplementary Table S4. The stacking ensemble is "
    "therefore best described as discriminating at least as well as the "
    "strongest individual learners rather than as uniformly superior: it "
    "was the best of the ten models in both validation cohorts, but in the "
    "external cohort it performed at the level of, and not meaningfully "
    "better than, the best base learners.")

rewrite_plain(P[63],
    "Calibration differed sharply between cohorts. In the internal test "
    "set the stacking ensemble was well calibrated overall (intercept "
    f"{CAL.loc['test','intercept']:+.3f}, 95% CI {CAL.loc['test','i_lo']:+.3f} "
    f"to {CAL.loc['test','i_hi']:+.3f}; slope {CAL.loc['test','slope']:.3f}, "
    f"95% CI {CAL.loc['test','s_lo']:.3f}-{CAL.loc['test','s_hi']:.3f}; "
    f"Brier score {CAL.loc['test','brier']:.3f}; E:O 1.00; Figure 4E): the "
    "calibration slope lay close to but statistically distinguishable from "
    "1, indicating mild over-confidence in the tails rather than near-"
    "perfect calibration, and internal calibration was clearly better than "
    "external. In eICU-CRD, calibration was poor: the intercept was "
    f"strongly negative ({CAL.loc['eICU','intercept']:+.3f}, 95% CI "
    f"{CAL.loc['eICU','i_lo']:+.3f} to {CAL.loc['eICU','i_hi']:+.3f}) and "
    f"the slope was substantially below unity ({CAL.loc['eICU','slope']:.3f}, "
    f"95% CI {CAL.loc['eICU','s_lo']:.3f}-{CAL.loc['eICU','s_hi']:.3f}), "
    f"with an expected-to-observed ratio of {CAL.loc['eICU','E_O']:.2f} and "
    f"a Brier score of {CAL.loc['eICU','brier']:.3f} (Figure 5D; all "
    "calibration metrics in Supplementary Table S5). In aggregate the "
    "model overestimated the number of NOAF events in the external cohort "
    "approximately threefold while still ranking patients in the correct "
    "order; this is a statement about cohort-level calibration, not about "
    "a uniform threefold inflation of every individual patient's risk.")

rewrite_plain(P[64],
    "Decision curve analysis showed that this distinction matters for "
    "threshold-based use. In the internal test set the net benefit of the "
    "stacking ensemble exceeded that of both treat-all and treat-none "
    "strategies across the analysed range of threshold probabilities (net "
    "benefit 0.220 at a threshold of 0.10, 0.174 at 0.20, 0.138 at 0.30, "
    "0.106 at 0.40 and 0.079 at 0.50; Figure 4D). In eICU-CRD the net "
    "benefit was positive only at threshold probabilities close to the "
    "local prevalence (0.004 at a threshold of 0.10) and became negative "
    "above that point (\u22120.025 at 0.20, \u22120.047 at 0.30, "
    "\u22120.061 at 0.40 and \u22120.071 at 0.50; Figure 5C; Supplementary "
    "Table S6), although it remained above the treat-all strategy at every "
    "threshold examined. These results indicate that the frozen model "
    "supports threshold-based decision-making in a population resembling "
    "the development cohort, but that its absolute predictions require "
    "recalibration, and its decision threshold re-derivation, before it is "
    "used in a cohort with a case mix resembling eICU-CRD. The in-sample "
    "training AUC of the stacking ensemble (0.842 with strictly "
    "out-of-fold predictions; 0.941 for the frozen pipeline refitted on "
    "the training set) is reported only to document that overfitting was "
    "limited, and no claim of improved generalisation rests on the "
    "training-to-test difference.")

# ---- SHAP results (dynamic values) ----
lab = dict(zip(RANK["Feature"], RANK["Label"]))
def sh(term):
    r = RANK[RANK.Feature == term].iloc[0]
    return (f"{r.MeanAbsSHAP:.3f}, 95% CI {r.CI_lo:.3f}-{r.CI_hi:.3f}")
top4 = RANK.sort_values("MeanAbsSHAP", ascending=False).head(4)["Feature"].tolist()
assert top4 == ["Age", "heart_rate_max", "congestive_heart_failure",
                "Vasopressors"], top4
rank_sd3 = STAB[STAB.Feature.isin(top4[:3])]["RankSD"].max()
r_stab = STAB[STAB.Feature.isin(["Age", "heart_rate_max",
                                 "congestive_heart_failure"])].sort_values(
    "MedianRank")
top10_max_dev = int(STAB.head(10)[["MedianRank", "MaxRank"]].assign(
    dev=lambda d: d.MaxRank - d.MedianRank)["dev"].max())

rewrite_plain(P[68],
    "Finally, we examined how the frozen ensemble reached its predictions, "
    "at both the population and the individual-patient level. In the SHAP "
    "summary beeswarm plot, a small set of predictors dominated the model's "
    f"decision-making (Figure 6A): age (mean |SHAP| {sh('Age')}), maximum "
    f"heart rate ({sh('heart_rate_max')}), congestive heart failure "
    f"({sh('congestive_heart_failure')}) and vasopressor exposure "
    f"({sh('Vasopressors')}) contributed most strongly, followed by weight "
    f"({sh('weight')}), minimum blood urea nitrogen ({sh('bun_min')}), "
    f"minimum heart rate ({sh('heart_rate_min')}) and male sex "
    f"({sh('Gender')}). The colour gradient indicated biologically "
    "coherent directions: older age, higher maximum and minimum heart "
    "rate, higher minimum blood urea nitrogen, the presence of congestive "
    "heart failure, vasopressor support, male sex and lower urine output "
    "and minimum sodium shifted the predicted probability towards NOAF. "
    "The ranking of the four leading predictors was reproduced in all ten "
    f"repeated subsamples (largest rank standard deviation {rank_sd3:.2f}), "
    "and every predictor among the top ten retained a maximum rank "
    f"deviation of {top10_max_dev} positions or fewer (Supplementary "
    "Figure S2).")

rewrite_plain(P[69],
    "Individual-patient waterfalls illustrated how population-level "
    "contributions aggregate into a single prediction (Figure 6B-E). For a "
    f"correctly classified patient with NOAF and a predicted probability "
    f"of {CASES.loc['TP','PredictedProbability']:.3f} (true positive), the "
    f"base value of {BASE_VAL:.3f} was raised chiefly by maximum heart "
    "rate, older age and congestive heart failure. For a correctly "
    "classified patient without NOAF and a predicted probability of "
    f"{CASES.loc['TN','PredictedProbability']:.3f} (true negative), the "
    "same base value was lowered principally by young age and by the "
    "absence of congestive heart failure and vasopressor exposure. The two "
    "discordant cases make the failure modes explicit: a patient without "
    "NOAF who received a predicted probability of "
    f"{CASES.loc['FP','PredictedProbability']:.3f} (false positive) was "
    "driven upwards by the same features that raise risk in true positives, "
    "and a patient with NOAF who received a predicted probability of "
    f"{CASES.loc['FN','PredictedProbability']:.3f} (false negative) lacked "
    "nearly all of them. The complete stacking model thus relies on "
    "clinically plausible risk factors, and its individual predictions can "
    "be traced to transparent feature contributions, with errors arising "
    "from patients whose risk profile resembles that of the opposite "
    "outcome group rather than from implausible model behaviour.")

print("Results done")

# =========================================================== 4. LEGENDS
rewrite_plain(P[44],
    "Figure 1. Flowchart of the study. The upper panel shows patient "
    "selection in the two databases and the numbers excluded at each step; "
    "the lower panels show the modelling workflow, in which univariable "
    "screening and LASSO selection were performed on the MIMIC-IV training "
    "set only and were re-run inside every cross-validation fold, base "
    "learners were ranked by training-set out-of-fold AUC and locked before "
    "the internal test set was opened, and the internal test set and the "
    "eICU-CRD cohort were examined only after the frozen final predictor "
    "had been fixed.")

surgical(P[54], "the dashed red line marks the one-standard-error (1-SE) "
                "solution (C = 0.0108, \u03bb = 92.5) and the dotted line "
                "the minimum-error solution (C = 0.441)",
         "the dashed red line marks the one-standard-error (1-SE) solution "
         "(C = 0.0108, \u03bb = 92.5) and the dotted line the "
         "minimum-error solution (C = 0.536)")
surgical(P[54], "retaining 25 variables", "retaining 26 variables")

rewrite_plain(P[59],
    "Figure 3. Performance of the nine candidate machine-learning models. "
    "(A) Receiver operating characteristic (ROC) curves and (B) "
    "precision-recall curves based on training-set out-of-fold predictions "
    "(n = 24,068), generated with fold-internal re-selection of predictors; "
    "(C) ROC curves and (D) precision-recall curves on the MIMIC-IV "
    "internal test set (n = 10,316). Panels show the nine pre-specified "
    "algorithms; the horizontal line in the precision-recall panels marks "
    "the event prevalence; AUCs carry DeLong 95% confidence intervals and "
    "APs bootstrap 95% confidence intervals (1,000 resamples), shown in the "
    "panel legends. Curves for the stacking ensemble and its base learners "
    "in the external cohort are shown in Figure 5, and all numeric "
    "estimates for every model and cohort are given in Table 2.")

rewrite_plain(P[66],
    "Figure 4. Construction and internal performance of the stacking "
    "ensemble. (A) Workflow: the univariable screen and the LASSO 1-SE rule "
    "were re-run inside every cross-validation fold, the nine candidate "
    "models were ranked by training-set out-of-fold AUC, and the three "
    "selected base learners (multilayer perceptron, random forest and "
    "XGBoost) were combined with a LASSO-penalised meta-learner into a "
    "single frozen predictor used for every evaluation and interpretation. "
    "(B) ROC, (C) precision-recall, (D) decision-curve and (E) calibration "
    "analysis of the stacking ensemble and its three base learners in the "
    "internal test set; colours and model order are identical in every "
    "panel, AUCs carry DeLong 95% confidence intervals and AP, net benefit "
    "and calibration metrics carry bootstrap 95% confidence intervals "
    "(1,000 resamples). In the calibration panel the diagonal represents "
    "perfect calibration; in the decision panel the grey lines represent "
    "the treat-all and treat-none strategies.")

rewrite_plain(P[71],
    "Figure 6. SHAP-based interpretation of the frozen stacking predictor. "
    "(A) Beeswarm summary of SHAP values for the 300 explained "
    "training-set patients on the probability scale; each point is one "
    "patient, the horizontal axis is the change in predicted probability of "
    "new-onset atrial fibrillation, and the colour encodes the feature "
    "value from low (blue) to high (red). Waterfall plots for the four "
    "pre-specified, extreme high-confidence case types: (B) true positive, "
    "(C) true negative, (D) false positive and (E) false negative. (F) "
    "Stability of the feature ranking across ten independent re-drawn "
    "explain sets (median rank with inter-run range; the tick marks the "
    "main analysis). In each waterfall the base value (mean model output) "
    "is shown at the bottom, individual feature contributions are added "
    "successively, and the final predicted probability is shown at the "
    "top; red bars increase and blue bars decrease the predicted "
    "probability.")

# ---- Table 2 caption
surgical(P[47], "Values are AUC or average precision (AP) with 95% "
                "confidence intervals from 1,000 stratified bootstrap "
                "resamples.",
         "Values are AUC or average precision (AP) with 95% confidence "
         "intervals (AUC: DeLong; AP: 1,000 stratified bootstrap "
         "resamples); the OOF AUCs underlying base-learner ranking are "
         "additionally reported to four decimal places in Supplementary "
         "Table S3.")

print("Legends done")

# =========================================================== 5. DISCUSSION
rewrite_plain(P[75],
    "In this study we developed and externally validated a stacking "
    "ensemble framework for predicting new-onset atrial fibrillation (NOAF) "
    "in critically ill patients, using a parsimonious 26-variable panel of "
    "routinely available demographic, comorbidity, intervention and "
    "first-24-hour physiological and laboratory variables, and evaluated "
    "its discrimination, calibration and interpretability across two large, "
    "independent critical-care databases. Four principal findings emerged. "
    "First, a stacking ensemble that combined three heterogeneous learners "
    "- the multilayer perceptron (MLP), the random forest (RF) and XGBoost "
    "- through a LASSO-penalised meta-learner discriminated at least as "
    "well as the strongest individual candidate models, both on the "
    "internal test set (AUC 0.843) and in the external cohort (0.753), and "
    "it was the single best model of the ten examined in both validation "
    "cohorts. Second, discrimination declined on cross-database validation "
    "(from 0.843 to 0.753) but remained moderate in a population with a "
    "nearly fourfold lower event rate, indicating that the learned "
    "predictor-outcome associations were largely though not completely "
    "preserved across healthcare systems. Third, and in contrast to "
    "discrimination, absolute risk was not transportable: the model was "
    "substantially better calibrated internally than externally but "
    "overestimated cohort-level risk roughly threefold in eICU-CRD "
    "(calibration slope 0.72, expected-to-observed ratio 3.02), so that "
    "discrimination transportability and absolute-risk transportability "
    "must be distinguished. Fourth, SHAP analysis showed that predictions "
    "were driven by a compact set of clinically coherent variables (age, "
    "maximum heart rate, congestive heart failure and vasopressor "
    "exposure), and that this ranking was stable across repeated "
    "subsamples; individual waterfall plots, including the two discordant "
    "cases, showed how risk estimates arise from additive contributions. "
    "Taken together, these findings indicate that heterogeneous stacking "
    "can yield NOAF risk estimates that are transportable in their ordering "
    "and transparent in their construction, while also demonstrating that a "
    "model of this kind is not transferable in its absolute predictions "
    "without recalibration.")

surgical(P[77], "(internal test AUC 0.833; external AUC 0.741)",
         "(internal test AUC 0.833; external AUC 0.738)")
surgical(P[77], "(ΔAUC 0.0025-0.0077) and absent in the external cohort "
                "against the strongest learners",
         "(\u0394AUC +0.0048 to +0.0084, all P < 0.001) and small and not "
         "uniformly significant in the external cohort (\u0394AUC +0.0013 "
         "vs. XGBoost, P = 0.373)")

surgical(P[79], "The decline in AUC from 0.846 to 0.757",
         "The decline in AUC from 0.843 to 0.753")
surgical(P[79], "(0.674 to 0.210)", "(0.668 to 0.202)")
surgical(P[79], "residual heterogeneity in how predictors and outcomes are "
                "recorded between the two databases",
         "residual heterogeneity in how predictors and outcomes are "
         "recorded and mapped between the two databases, including "
         "exposures such as antibiotics that are documented much more "
         "completely in MIMIC-IV")

surgical(P[80], "in eICU-CRD the calibration slope was 0.72 and the "
                "expected-to-observed ratio 3.06, so a predicted "
                "probability of 0.30 corresponded to an observed event "
                "rate of roughly 0.10, and the corresponding decision "
                "curve showed no net benefit above that threshold",
         "in eICU-CRD the calibration slope was 0.72 and the "
         "expected-to-observed ratio 3.02, indicating a roughly threefold "
         "cohort-level overestimation of event rates, and the "
         "corresponding decision curve showed no net benefit above the "
         "local prevalence")

# MO11 note in Results 3.1
surgical(P[41], "and antibiotic exposure much less frequent (48.3% vs. "
                "82.7%; SMD 0.78).",
         "and antibiotic exposure much less frequent (48.3% vs. 82.7%; SMD "
         "0.78); part of this difference plausibly reflects database-"
         "specific documentation and harmonisation of antibiotic exposure "
         "rather than case mix alone, as considered in the Discussion.")

# strengths [91]
rewrite_plain(P[91],
    "This study has several strengths. It combines heterogeneous base "
    "learners in a stacking ensemble validated across two independent, "
    "large critical-care databases; it relies on a parsimonious 26-variable "
    "panel composed entirely of variables routinely recorded within the "
    "first 24 hours, favouring feasibility at the bedside; its development "
    "protocol protected against leakage at every stage, because imputation "
    "was fitted on training rows only after the 70:30 split, the "
    "univariable screen and the LASSO were re-run inside every "
    "cross-validation fold, base learners were ranked by training-set "
    "out-of-fold performance and locked before the internal test set was "
    "opened, and one frozen predictor generated all reported performance "
    "estimates and all interpretations; and its evaluation included "
    "calibration, decision curve analysis with explicit loss-ratio "
    "interpretation, and interpretation at both population and "
    "individual-patient levels, including discordant cases.")

# limitations [92] — MO02 + single-imputation note
surgical(P[92], "so that the cohort is enriched for patients with an "
                "observable outcome and the reported discrimination may be "
                "optimistic for an unselected ICU population.",
         "so that the cohort is enriched for patients with an observable "
         "outcome and the reported discrimination may be optimistic for an "
         "unselected ICU population; the model is accordingly applicable "
         "only to patients who remain in the ICU beyond the first 48 hours, "
         "and a 24-hour landmark analysis in the broader admission "
         "population is a natural next step.")
surgical(P[92], "Future work should therefore prioritise",
         "The single completed dataset per split likewise implies that the "
         "confidence intervals reported here do not reflect imputation "
         "uncertainty; given the low missingness of the retained variables "
         "this is unlikely to alter the conclusions. Future work should "
         "therefore prioritise")

surgical(P[94], "calibration was close to ideal in the internal test set "
                "but poor in the external cohort",
         "calibration was substantially better in the internal test set "
         "than in the external cohort")

print("Discussion done")

# =========================================================== 6. TABLE 2
MODELS = ["Logistic", "Ridge", "DT", "KNN", "SVM", "MLP", "RF", "LightGBM",
          "XGBoost", "Stacking"]
tbl = doc.tables[1]
# header second row: "AP" -> "AP (95% CI)"
hdr = tbl.rows[1]
for c in hdr.cells:
    if c.text.strip() == "AP":
        c.text = "AP (95% CI)"
for i, m in enumerate(MODELS, start=2):
    row = tbl.rows[i]
    vals = [fmt(m, "train"), fmt(m, "train", col="AP"),
            fmt(m, "test"), fmt(m, "test", col="AP"),
            fmt(m, "eICU"), fmt(m, "eICU", col="AP")]
    for j, v in enumerate(vals, start=1):
        cell = row.cells[j]
        # keep first paragraph formatting
        par = cell.paragraphs[0]
        for r in list(par.runs)[1:]:
            r._element.getparent().remove(r._element)
        if par.runs:
            par.runs[0].text = v
        else:
            par.add_run(v)
print("Table 2 rebuilt")

# =========================================================== 7. FIGURES
def fig_width(p):
    ext = p._element.findall(".//" + qn("wp:extent"))
    return int(ext[0].get("cx")) if ext else Emu(int(15.5 * 360000))

def replace_picture(p, img):
    w = fig_width(p)
    for r in list(p.runs):
        if r._element.findall(".//" + qn("w:drawing")):
            r._element.getparent().remove(r._element)
    run = p.add_run()
    run.add_picture(str(img), width=Emu(w))

replace_picture(P[43], FIG / "Figure_Flowchart_study.png")
replace_picture(P[53], FIG / "Fig2_Screening.png")
replace_picture(P[58], FIG / "Fig3_NineModels_TrainTest.png")
replace_picture(P[65], FIG / "Figure4_stacking_internal.png")
replace_picture(P[70], FIG / "Figure6_SHAP.png")

# insert Figure 5 image + legend after the Figure 4 legend (P[66])
anchor = P[67]._p  # "3.5 Model interpretation..." heading element
from docx.oxml import OxmlElement  # noqa: E402

def new_paragraph_before(anchor_el):
    el = OxmlElement("w:p")
    anchor_el.addprevious(el)
    from docx.text.paragraph import Paragraph
    return Paragraph(el, P[67]._parent)

img5 = new_paragraph_before(anchor)
img5.alignment = WD_ALIGN_PARAGRAPH.CENTER
img5.add_run().add_picture(str(FIG / "Figure5_stacking_external.png"),
                           width=Emu(fig_width(P[65])))

leg5 = new_paragraph_before(anchor)
r0 = leg5.add_run("Figure 5. ")
r0.bold = True
leg5.add_run(
    "External validation of the stacking ensemble in the eICU-CRD cohort. "
    "(A) ROC, (B) precision-recall, (C) decision-curve and (D) calibration "
    "analysis of the frozen stacking predictor and its three base learners "
    "(n = 32,067; NOAF prevalence 7.7%). Confidence intervals as in Figure "
    "4. Calibration is visibly poorer than in the internal test set, with a "
    "substantially subunit slope and a strongly negative calibration "
    "intercept.")

# =========================================================== 8. DELETE MA05 dups
for idx in (87, 86, 85, 84):  # delete from the end to keep refs valid
    p = P[idx]
    p._element.getparent().remove(p._element)
print("deleted duplicated discussion paragraphs (84-87)")

doc.save(str(DOC))
print("[SAVED]", DOC)
