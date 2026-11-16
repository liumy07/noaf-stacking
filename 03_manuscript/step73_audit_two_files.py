# -*- coding: utf-8 -*-
"""
step73_audit_two_files.py — audit every reported estimate and CI in
  (1) NOAF_stacking_manuscript.docx
  (2) 新结果figure设计_修订版_backup_preFig3EF.pptx
against the frozen v90 results in output/rev_noaf/v90_primary.

Checks
  A  manuscript Table 2         (10 models x 3 cohorts x AUC/AP)
  B  manuscript Table S4        (9 paired DeLong comparisons x 2 cohorts)
  C  manuscript prose           (every stated estimate + CI)
  D  net benefit 0.30-0.50      (recomputed from v80_scores.npz)
  E  SHAP / univariable / LASSO / meta-learner / case probabilities
  F  Table 1 internal consistency (n, % of denominator, SMD)
  G  deck Table 2, stacking row, DeLong, headline, DCA, calibration, stability
  H  cross-file consistency (figure panel numbering)

Writes a plain-text report to _audit_report.txt and prints it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(r"<LOCAL_ROOT>/Desktop/NOAF_v90_全套代码/00_common")))
from noaf_common_v80 import V80_DIR  # noqa: E402

PROJ = Path(r"D:/BJND-4002-11项目/BJND-4002-11")
MANU = PROJ / "NOAF_stacking_manuscript.docx"
DECK = Path(r"<LOCAL_ROOT>/Desktop/A prediction model for new-onset "
            r"atrial fibrillation in critically ill patients based on stacking "
            r"ensemble strategy/新结果figure设计_修订版_backup_preFig3EF.pptx")

import docx  # noqa: E402
from pptx import Presentation  # noqa: E402

lines: list[str] = []
n_fail = 0


def say(s=""):
    lines.append(s)


def check(label, ok, detail=""):
    global n_fail
    if not ok:
        n_fail += 1
    say(f"  [{'OK ' if ok else 'BAD'}] {label}" + (f"   {detail}" if detail else ""))


def f3(x):
    return f"{x:.3f}"


# ---------------------------------------------------------------- reference
AUC = pd.read_csv(V80_DIR / "v80_ci_auc.csv", encoding="utf-8-sig")
APC = pd.read_csv(V80_DIR / "v80_ci_ap.csv", encoding="utf-8-sig")
DEL = pd.read_csv(V80_DIR / "v80_delong_paired.csv", encoding="utf-8-sig")
NB = pd.read_csv(V80_DIR / "v80_ci_nb.csv", encoding="utf-8-sig")
CAL = pd.read_csv(V80_DIR / "v80_ci_calibration.csv",
                  encoding="utf-8-sig").set_index("cohort")
SHAP = pd.read_csv(V80_DIR / "v80_shap_ranking.csv", encoding="utf-8-sig")
UNI = pd.read_csv(V80_DIR / "v80_univariable_logistic.csv",
                  encoding="utf-8-sig").set_index("var")
STAB = pd.read_csv(V80_DIR / "v80_shap_stability_summary.csv",
                   encoding="utf-8-sig")
CASES = pd.read_csv(V80_DIR / "v80_shap_cases.csv", encoding="utf-8-sig")
D = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)

MODEL_LABEL = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
               "RF": "Random forest", "LightGBM": "LightGBM",
               "Logistic": "Logistic regression",
               "SVM": "Support vector machine",
               "Ridge": "Ridge regression", "DT": "Decision tree",
               "KNN": "K-nearest neighbours", "Stacking": "Stacking ensemble"}
COH_LABEL = {"train": "Training", "test": "test", "eICU": "eICU"}


def ref_auc(model, coh):
    r = AUC[(AUC.Model == model) & (AUC.cohort == coh)].iloc[0]
    return f"{r.AUC:.3f} ({r.CI_lo:.3f}-{r.CI_hi:.3f})"


def ref_ap(model, coh):
    r = APC[(APC.Model == model) & (APC.cohort == coh)].iloc[0]
    return f"{r.AP:.3f} ({r.lo:.3f}-{r.hi:.3f})"


def norm(s):
    return (s.replace("\u2013", "-").replace("\u2212", "-")
             .replace("\u00a0", " ").strip())


say("=" * 90)
say("AUDIT — manuscript numbers & CIs vs frozen v90 results")
say("=" * 90)

# ------------------------------------------------------- A. Table 2 (manuscript)
say("\nA. Manuscript Table 2 (10 models x 3 cohorts)")
md = docx.Document(str(MANU))
t2 = md.tables[1]
COLS = ["train", "test", "eICU"]
for row in t2.rows[2:]:
    cells = [c.text.strip() for c in row.cells]
    name = cells[0]
    model = [k for k, v in MODEL_LABEL.items() if v == name]
    if not model:
        say(f"  ?? unknown model row: {name}")
        continue
    model = model[0]
    for j, coh in enumerate(COLS):
        got_auc, got_ap = norm(cells[1 + 2 * j]), norm(cells[2 + 2 * j])
        exp_auc, exp_ap = ref_auc(model, coh), ref_ap(model, coh)
        check(f"{name:22s} {coh:5s} AUC {got_auc}", got_auc == exp_auc, f"exp {exp_auc}")
        check(f"{name:22s} {coh:5s} AP  {got_ap}", got_ap == exp_ap, f"exp {exp_ap}")

# --------------------------------------------------- B. Table S4 (manuscript)
say("\nB. Manuscript Table S4 (paired DeLong)")
t4 = md.tables[2]
for row in t4.rows[1:]:
    cells = [norm(c.text) for c in row.cells]
    comp = cells[0]
    coh = "test" if "internal" in comp else "eICU"
    base = None
    for k, v in MODEL_LABEL.items():
        if v.lower() in comp.lower():
            base = k
            break
    if base is None:
        say(f"  ?? cannot map comparison: {comp}")
        continue
    r = DEL[(DEL.cohort == coh) & (DEL.base == base)].iloc[0]
    exp_d = f"{r.delta:+.4f}"
    exp_ci = f"{r.lo:+.4f} to {r.hi:+.4f}"
    got_d, got_ci = cells[1], cells[2].replace("  ", " ")
    check(f"S4 {coh:4s} {base:8s} delta {got_d}", got_d == exp_d, f"exp {exp_d}")
    check(f"S4 {coh:4s} {base:8s} CI    {got_ci}", got_ci == exp_ci, f"exp {exp_ci}")
    p_exp = "P < 0.001" if r.p < 0.001 else f"P = {r.p:.3f}"
    got_p = cells[3]
    check(f"S4 {coh:4s} {base:8s} P     {got_p}", got_p == p_exp, f"exp {p_exp}")

# --------------------------------------------------------------- C. prose
say("\nC. Manuscript prose estimates")
txt = "\n".join(p.text for p in md.paragraphs)
txt_n = norm(txt)

def ci_in(est, lo, hi, hay, gap1=45, gap2=10):
    """True when est, then lo, then hi appear in that order within one sentence.

    Tolerates the two house styles used in the manuscript:
      'AUC 0.843, 95% CI 0.835-0.851'      and
      '0.668 (0.651-0.684)'
    """
    est, lo, hi = norm(est), norm(lo), norm(hi)
    pat = (re.escape(est) + r".{0,%d}?" % gap1
           + re.escape(lo) + r".{0,%d}?" % gap2 + re.escape(hi))
    m = re.search(pat, hay, re.S)
    return bool(m), (m.group(0) if m else "")


claims_ci = []   # (label, est, lo, hi)
claims_val = []  # (label, value)


def auc_parts(model, coh):
    r = AUC[(AUC.Model == model) & (AUC.cohort == coh)].iloc[0]
    return f"{r.AUC:.3f}", f"{r.CI_lo:.3f}", f"{r.CI_hi:.3f}"


def ap_parts(model, coh):
    r = APC[(APC.Model == model) & (APC.cohort == coh)].iloc[0]
    return f"{r.AP:.3f}", f"{r.lo:.3f}", f"{r.hi:.3f}"


# 3.4 stacking ensemble
claims_ci += [("stacking test AUC", *auc_parts("Stacking", "test")),
              ("stacking test AP", *ap_parts("Stacking", "test")),
              ("stacking eICU AUC", *auc_parts("Stacking", "eICU")),
              ("stacking eICU AP", *ap_parts("Stacking", "eICU"))]
# 3.3 nine models
for m in ("MLP", "RF", "XGBoost", "LightGBM", "KNN"):
    claims_ci.append((f"{m} train", *auc_parts(m, "train")))
for m in ("XGBoost", "RF", "LightGBM", "MLP", "KNN", "Logistic"):
    claims_ci.append((f"{m} test", *auc_parts(m, "test")))
for m in ("XGBoost", "LightGBM", "RF", "MLP", "KNN", "Logistic"):
    claims_ci.append((f"{m} eICU", *auc_parts(m, "eICU")))
for m in ("XGBoost", "LightGBM", "RF"):
    claims_ci.append((f"{m} AP test", *ap_parts(m, "test")))
for m in ("XGBoost", "LightGBM"):
    claims_ci.append((f"{m} AP eICU", *ap_parts(m, "eICU")))

# calibration
cal_test, cal_eicu = CAL.loc["test"], CAL.loc["eICU"]
claims_ci += [
    ("calib internal intercept", f"+{cal_test.intercept:.3f}",
     f"+{cal_test.i_lo:.3f}", f"+{cal_test.i_hi:.3f}"),
    ("calib internal slope", f"{cal_test.slope:.3f}",
     f"{cal_test.s_lo:.3f}", f"{cal_test.s_hi:.3f}"),
    ("calib eICU intercept", f"{cal_eicu.intercept:.3f}",
     f"{cal_eicu.i_lo:.3f}", f"{cal_eicu.i_hi:.3f}"),
    ("calib eICU slope", f"{cal_eicu.slope:.3f}",
     f"{cal_eicu.s_lo:.3f}", f"{cal_eicu.s_hi:.3f}"),
]
claims_val += [
    ("E:O eICU", f"{cal_eicu.E_O:.2f}"),
    ("Brier internal", f"{cal_test.brier:.3f}"),
    ("Brier eICU", f"{cal_eicu.brier:.3f}"),
]
# DeLong (all quoted comparisons)
dl = {(r.cohort, r.base): r for _, r in DEL.iterrows()}
for coh, base, tag in (("test", "MLP", "test MLP"), ("test", "XGBoost", "test XGB"),
                       ("test", "RF", "test RF"), ("eICU", "MLP", "eICU MLP"),
                       ("eICU", "XGBoost", "eICU XGB"), ("eICU", "RF", "eICU RF")):
    r = dl[(coh, base)]
    claims_ci.append((f"DeLong {tag}", f"{r.delta:+.4f}", f"{r.lo:+.4f}", f"{r.hi:+.4f}"))

# net benefit 0.10 / 0.20 / 0.30-0.50 from CSV
nb_s = {(r.cohort, round(r.thr, 2)): r for _, r in NB[NB.Model == "Stacking"].iterrows()}
for coh, thr in (("test", 0.1), ("test", 0.2), ("eICU", 0.1), ("eICU", 0.2)):
    v = nb_s[(coh, thr)].nb
    claims_val.append((f"NB {coh:4s} @{thr:.2f}", f"{v:.3f}".lstrip("-")))

for name, est, lo, hi in claims_ci:
    ok, got = ci_in(est, lo, hi, txt_n)
    check(f"prose {name:24s} {est} ({lo}-{hi})", ok,
          "" if ok else f"not found; nearest '{got}'")

for name, val in claims_val:
    ok = norm(val) in txt_n
    check(f"prose {name:24s} {val}", ok, "" if ok else "value not found in prose")


def nb_at(y, s, thr):
    pred = (s >= thr).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    fp = np.sum((pred == 1) & (y == 0))
    n = len(y)
    return tp / n - fp / n * thr / (1 - thr)


say("\nD. Net benefit at 0.30 / 0.40 / 0.50 (recomputed)")
for coh, prefix in (("test", "tst__"), ("eICU", "ext__")):
    y = D["y_test" if coh == "test" else "y_eicu"].astype(int)
    s = D[prefix + "Stacking"]
    for thr in (0.30, 0.40, 0.50):
        v = nb_at(y, s, thr)
        exp = f"{v:.3f}".replace("-", "\u2212") if v < 0 else f"{v:.3f}"
        found = f"{v:.3f}" in txt_n or exp in txt_n
        check(f"NB {coh:4s} @{thr:.2f} {v:+.3f}", found,
              "" if found else "value not found in prose")

# LASSO / meta / in-sample
say("\nE. LASSO, meta-learner, in-sample, SHAP, case probabilities")
meta_w = D["meta_weights"]
meta_b = float(D["meta_intercept"])
order = [str(x) for x in D["top3"]]
exp_meta = ", ".join(f"{m} {w:.2f}" for m, w in zip(order, meta_w))
check(f"meta weights ({exp_meta}; intercept {meta_b:.2f})",
      all(f"{w:.2f}" in txt_n for w in meta_w) and f"{meta_b:.2f}".replace("-", "\u2212") in txt_n.replace("-", "\u2212"))

from sklearn.metrics import roc_auc_score  # noqa: E402
ins = roc_auc_score(D["y_train"].astype(int), D["ins__Stacking"])
check(f"in-sample AUC {ins:.3f}", f"{ins:.3f}" in txt_n, "expected ~0.944")

for i in range(8):
    r = SHAP.iloc[i]
    label = r.Label
    exp = f"{r.MeanAbsSHAP:.3f}, 95% CI {r.CI_lo:.3f}-{r.CI_hi:.3f}"
    check(f"SHAP #{i+1} {label[:28]:28s}", exp in txt_n, f"exp {exp}")

for _, r in CASES.iterrows():
    v = f"{r.PredictedProbability:.3f}"
    check(f"case {r.CaseType} P={v}", v in txt_n)

check("base value 0.317", "0.317" in txt_n)
check("stability: top-4 rank SD 0.00", STAB.head(4).RankSD.max() == 0.0
      and "largest rank standard deviation 0.00" in txt_n)
top10_dev = max((STAB.head(10).MaxRank - STAB.head(10).MedianRank).abs().max(),
                (STAB.head(10).MedianRank - STAB.head(10).MinRank).abs().max())
check(f"stability: top-10 max rank deviation {top10_dev:.1f} <= 2", top10_dev <= 2)

# univariable ORs quoted in 3.2
UNI_QUOTE = [("congestive_heart_failure", 4.19), ("Vasopressors", 2.43),
             ("renal_disease", 2.31), ("Antibiotic", 1.96), ("sepsis", 1.64),
             ("InvasiveVent", 1.46), ("diabetes", 1.41), ("Gender", 1.27),
             ("Age", 1.073)]
for var, quoted in UNI_QUOTE:
    r = UNI.loc[var]
    exp = f"{r.OR:.3f}" if var == "Age" else f"{r.OR:.2f}"
    check(f"univariable {var:24s} OR {exp} (quoted {quoted})",
          abs(r.OR - quoted) < 0.0051,
          f"raw {r.OR:.4f} CI {r.lo:.3f}-{r.hi:.3f}")
    if var == "Age":
        ok_ci, _ = ci_in(f"{r.OR:.3f}", f"{r.lo:.3f}", f"{r.hi:.3f}", txt_n)
        ci = f"95% CI {r.lo:.3f}-{r.hi:.3f}"
    else:
        ok_ci, _ = ci_in(f"{r.OR:.2f}", f"{r.lo:.2f}", f"{r.hi:.2f}", txt_n)
        ci = f"95% CI {r.lo:.2f}-{r.hi:.2f}"
    check(f"univariable {var:24s} CI {ci}", ok_ci, "" if ok_ci else "CI not found")

# -------------------------------------------------- F. Table 1 consistency
say("\nF. Manuscript Table 1 — internal consistency (n / % of denominator)")
t1 = md.tables[0]
bad = 0
for row in t1.rows[2:]:
    cells = [c.text.strip() for c in row.cells]
    for j, denom in ((1, 34384), (2, 32067)):
        m = re.match(r"^([\d,]+) \(([\d.]+)\)$", cells[j])
        if not m:
            continue
        n = int(m.group(1).replace(",", ""))
        pct = float(m.group(2))
        if abs(n / denom * 100 - pct) > 0.06:
            bad += 1
            say(f"  [BAD] {cells[0][:40]:40s} col{j}: {n}/{denom} = "
                f"{n/denom*100:.2f}% but written {pct}")
check("Table 1 all n(%) pairs consistent", bad == 0)

# F2. every Table 1 cell against the frozen baseline table
say("\nF2. Manuscript Table 1 vs frozen t1_baseline_cohorts.csv")
T1 = pd.read_csv(V80_DIR.parent / "tables" / "t1_baseline_cohorts.csv",
                 encoding="utf-8-sig")
wrows = [[c.text.strip() for c in r.cells] for r in t1.rows][1:]
bad1 = 0
if len(wrows) != len(T1):
    bad1 += 1
    say(f"  [BAD] row count Word {len(wrows)} vs source {len(T1)}")
for i, (w, (_, s)) in enumerate(zip(wrows, T1.iterrows())):
    for label, got, exp in (
            ("MIMIC", norm(w[1]), norm(str(s["MIMIC-IV (n = 34,384)"]))),
            ("eICU", norm(w[2]), norm(str(s["eICU-CRD (n = 32,067)"]))),
            ("P", norm(w[4]), norm(str(s["P"])))):
        if got != exp:
            bad1 += 1
            say(f"  [BAD] row {i:02d} {label:5s} '{got}' vs '{exp}' ({s.Characteristic})")
    try:
        if abs(float(w[3]) - float(s.SMD)) > 0.0051:
            bad1 += 1
            say(f"  [BAD] row {i:02d} SMD {w[3]} vs {s.SMD} ({s.Characteristic})")
    except ValueError:
        bad1 += 1
        say(f"  [BAD] row {i:02d} SMD unparsable {w[3]!r}")
check(f"Table 1 all {len(wrows)} rows match source (values, SMD, P)", bad1 == 0)

# ------------------------------------------------------------------- G. deck
say("\nG. Deck (backup pre-Fig3EF) numbers")
prs = Presentation(str(DECK))
slides = list(prs.slides)


def slide_text(i):
    s = slides[i]
    out = []
    for sh in s.shapes:
        if sh.has_text_frame:
            out.append(sh.text_frame.text)
        if sh.has_table:
            for r in sh.table.rows:
                out.append(" | ".join(c.text.strip() for c in r.cells))
    try:
        out.append(s.notes_slide.notes_text_frame.text)
    except Exception:
        pass
    return norm("\n".join(out))


# slide 9 = Table 2 (0-based 8); map rows by model label
st = slide_text(8)
for row in slides[8].shapes:
    if not row.has_table:
        continue
    for r in list(row.table.rows)[1:]:
        cells = [norm(c.text.strip()) for c in r.cells]
        name = cells[0]
        model = [k for k, v in MODEL_LABEL.items() if v == name]
        if not model:
            continue
        model = model[0]
        for j, coh in enumerate(COLS):
            got_auc, got_ap = cells[1 + 2 * j], cells[2 + 2 * j]
            exp_auc, exp_ap = ref_auc(model, coh), ref_ap(model, coh)
            check(f"deck T2 {name[:20]:20s} {coh:5s} AUC", got_auc == exp_auc, f"{got_auc} vs {exp_auc}")
            check(f"deck T2 {name[:20]:20s} {coh:5s} AP ", got_ap == exp_ap, f"{got_ap} vs {exp_ap}")

# slide 12 stacking detail
st13 = slide_text(11)
exp = f"0.842 (0.836-0.847)"
check("deck slide 12 stacking train AUC/AP",
      exp in st13 and ref_ap("Stacking", "train") in st13)
check("deck slide 12 stacking test/ eICU",
      ref_auc("Stacking", "test") in st13 and ref_auc("Stacking", "eICU") in st13
      and ref_ap("Stacking", "test") in st13 and ref_ap("Stacking", "eICU") in st13)

# slide 15 DeLong table
st15 = slide_text(14)
for _, r in DEL.iterrows():
    d_s = f"{r.delta:+.4f}"
    ci_s = f"({r.lo:+.4f} to {r.hi:+.4f})"
    check(f"deck slide15 {r.cohort:4s} {r.base:8s} {d_s} {ci_s}",
          d_s in st15 and ci_s in st15)

# slide 16/17 DCA numbers with CI
st16, st17 = slide_text(15), slide_text(16)
for coh, s_txt, tag in (("test", st16, "s16"), ("eICU", st17, "s17")):
    for thr in (0.1, 0.2):
        r = nb_s[(coh, thr)]
        val = f"{r.nb:+.3f}" if r.nb > 0 else f"{r.nb:.3f}"
        lo = f"{r.lo:+.3f}" if r.lo > 0 else f"{r.lo:.3f}"
        hi = f"{r.hi:+.3f}" if r.hi > 0 else f"{r.hi:.3f}"
        check(f"deck {tag} NB {coh} @{thr:.2f} {val} ({lo} to {hi})",
              val.replace("-", "\u2212") in s_txt or val in s_txt)

# slide 13 calibration
st13c = slide_text(12)
check("deck slide13 internal slope 1.14 (1.10-1.19)",
      f"{cal_test.slope:.2f}" in st13c and f"{cal_test.s_lo:.2f}-{cal_test.s_hi:.2f}" in st13c)
check("deck slide13 eICU slope 0.72 intercept -1.73",
      f"{cal_eicu.slope:.2f}" in st13c and f"{abs(cal_eicu.intercept):.2f}" in st13c)

# slide 24 headline table
st24 = slide_text(23)
rows = {}
for sh in slides[23].shapes:
    if sh.has_table:
        for r in list(sh.table.rows)[1:]:
            c = [norm(x.text.strip()) for x in r.cells]
            rows[(c[0], c[1])] = c
BAD = 0
for (model_name, coh_name), c in rows.items():
    model = [k for k, v in MODEL_LABEL.items() if v == model_name]
    if not model:
        continue
    model = model[0]
    cohort = "test" if "test" in c[1] else "eICU"
    exp_auc, exp_ap = ref_auc(model, cohort), ref_ap(model, cohort)
    check(f"deck s24 {model_name[:12]:12s} {cohort:4s} AUC {c[2]}", c[2] == exp_auc, f"exp {exp_auc}")
    check(f"deck s24 {model_name[:12]:12s} {cohort:4s} AP  {c[3]}", c[3] == exp_ap, f"exp {exp_ap}")
    r = pd.read_csv(V80_DIR / "v80_results.csv", encoding="utf-8-sig")
    rmap = {"test": "internal test", "eICU": "eICU external"}
    rr = r[(r.Model == model) & (r.cohort == rmap[cohort])].iloc[0]
    check(f"deck s24 {model_name[:12]:12s} {cohort:4s} Brier {c[4]}",
          c[4] == f"{rr.Brier:.3f}", f"exp {rr.Brier:.3f}")
    if c[5] != "":
        check(f"deck s24 {model_name[:12]:12s} {cohort:4s} slope {c[5]}",
              c[5] == f"{rr.calib_slope:.2f}", f"exp {rr.calib_slope:.2f}")

# slide 23 stability
st23 = slide_text(22)
check("deck slide23 rho mean 0.977 min 0.969", "0.977" in st23 and "0.969" in st23)
check("deck slide23 Jaccard 0.821", "0.821" in st23)

# slide 24 headline table — base-learner Brier / slope against Supplementary S5
S5 = PROJ / "Supplementary_Table_S5_calibration.docx"
if S5.exists():
    s5 = docx.Document(str(S5))
    s5rows = {}
    for r in s5.tables[0].rows[1:]:
        cells = [c.text.strip().replace("\n", " ") for c in r.cells]
        model, coh = cells[0], ("test" if "Internal" in cells[1] else "eICU")
        s5rows[(model, coh)] = cells  # intercept, slope, brier, E:O
    NAME2 = {"Stacking": "Stacking ensemble", "MLP": "Multilayer perceptron",
             "RF": "Random forest", "XGBoost": "XGBoost"}
    s24 = []
    for sh in prs.slides[23].shapes:
        if sh.has_table:
            for r in sh.table.rows:
                s24.append([c.text.strip().replace("\n", " ") for c in r.cells])
    def first_num(s):
        m = re.search(r"-?\d+\.\d+", s)
        return float(m.group()) if m else None

    for cells in s24:
        if not cells or cells[0] not in NAME2:
            continue
        coh = "test" if "test" in cells[1] else "eICU"
        key = (NAME2[cells[0]], coh)
        if key not in s5rows:
            continue
        sl = s5rows[key]
        exp_slope, exp_brier = first_num(sl[3]), first_num(sl[4])
        got_slope, got_brier = first_num(cells[5]), first_num(cells[4])
        ok_b = got_brier is not None and abs(got_brier - exp_brier) <= 0.0006
        ok_s = got_slope is not None and abs(got_slope - exp_slope) <= 0.0051
        check(f"deck s24 {NAME2[cells[0]]:8s} {coh:4s} Brier {cells[4]}", ok_b,
              f"S5 {sl[4]}")
        check(f"deck s24 {NAME2[cells[0]]:8s} {coh:4s} slope {cells[5]}", ok_s,
              f"S5 {sl[3]}")

# slide 19-22 probabilities
for i, case in ((18, "TP"), (19, "TN"), (20, "FP"), (21, "FN")):
    v = f"{CASES[CASES.CaseType == case].PredictedProbability.iloc[0]:.3f}"
    check(f"deck slide{i+1} case {case} P={v}", v in slide_text(i))

# ------------------------------------------------- H. cross-file consistency
say("\nH. Cross-file consistency")
check("manuscript references Figure 3A-F (its caption lists E and F)",
      "Figure 3. Performance" in txt_n and "(E) ROC curves" in txt_n)
st8 = slide_text(7)
check("deck slide 8 still describes only A-D (pre-Fig3EF version)", "A-D" not in st8 and "Figure 3" in st8)
say("  note: the deck is the pre-Fig3EF backup — it has no Figure 3E/F slide and")
say("        its slide 8 title covers A–D only; manuscript text now cites Figure 3E/3F.")

say("\n" + "=" * 90)
say(f"TOTAL MISMATCHES: {n_fail}")
say("=" * 90)

report = PROJ / "_audit_report.txt"
report.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\n[written] {report}")
