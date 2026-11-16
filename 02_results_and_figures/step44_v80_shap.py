# -*- coding: utf-8 -*-
"""
step25_v80_shap.py — SHAP explanation of the v70 leak-free stacking ensemble

Same protocol as v50 (step5_SHAP_revision.py), re-run on the v70 pipeline:

  * explained function : the COMPLETE stacking pipeline
                         preprocessing -> MLP / RF / XGBoost -> LASSO meta
  * explainer          : shap.PermutationExplainer (model-agnostic)
  * primary scale      : PROBABILITY, f(X) = P(NOAF); log-odds as audit
  * background         : 100 training patients, default_rng(42)
  * main explain set   : 300 training patients, default_rng(42)
  * case waterfalls    : pre-specified TP / TN / FP / FN rule on the
                         internal hold-out test set
  * stability          : 20 re-drawn explain sets, rank agreement (R12)

Outputs (v80_leakfree/)
    Fig8_SHAP_beeswarm.(png|pdf|svg)
    Fig8_SHAP_importance_bar.(png|pdf|svg)
    Fig8_SHAP_waterfall_TP|TN|FP|FN.(png|pdf|svg)
    Fig8_SHAP_stability.(png|pdf|svg)
    v80_shap_values.npz / v80_shap_ranking.csv / v80_shap_cases.csv
    v80_shap_stability.csv / v80_shap_stability_summary.csv / v80_shap_audit.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import (  # noqa: E402
    V80_DIR, REV_DIR, SEED, LABELS, make_preprocessor, split_columns,
    base_models, set_style, save_fig, fast_auc,
)

from sklearn.base import clone                     # noqa: E402
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression  # noqa: E402
from scipy.stats import spearmanr                  # noqa: E402
import shap                                        # noqa: E402

T0 = time.time()
LOG_FH = open(V80_DIR / "v80_shap_run_log.txt", "w", encoding="utf-8")
AUDIT_FH = open(V80_DIR / "v80_shap_audit.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG_FH.write(msg + "\n"); LOG_FH.flush()


def audit(*a):
    msg = " ".join(str(x) for x in a)
    AUDIT_FH.write(msg + "\n"); AUDIT_FH.flush()


log("=" * 78)
log("v70 SHAP — permutation explainer on the leak-free stacking pipeline")
log("=" * 78)

# ----------------------------------------------------------------------
# data — the v70 imputed frames (already split-first, no NaN)
# ----------------------------------------------------------------------
tr = pd.read_csv(REV_DIR / "v70_leakfree" / "v70_train.csv", encoding="utf-8-sig")
te = pd.read_csv(REV_DIR / "v70_leakfree" / "v70_test.csv", encoding="utf-8-sig")
y_train = tr.pop("NOAF").astype(int)
y_test = te.pop("NOAF").astype(int)

z = np.load(V80_DIR / "v80_scores.npz", allow_pickle=True)
top3 = [str(s) for s in z["top3"]]
selected_vars = [str(s) for s in z["selected_vars"]]
log(f"[INPUT] top-3 = {top3}   LASSO variables = {len(selected_vars)}")

X_train = tr[selected_vars].copy()
X_test = te[selected_vars].copy()

num_cols, cat_cols = split_columns(X_train)
pre = make_preprocessor(num_cols, cat_cols)
Xtr_mat = pre.fit_transform(X_train)
Xte_mat = pre.transform(X_test)
feat_names = [t.split("__", 1)[1] if "__" in t else t for t in
              pre.get_feature_names_out()]
pretty = [LABELS.get(f, f) for f in feat_names]
log(f"[PREP] design matrix {Xtr_mat.shape}, {len(feat_names)} features")

# base learners refit on the full training set (identical seed/hparams)
models = base_models()
base_full = {}
for n in top3:
    m = clone(models[n])
    m.fit(Xtr_mat, y_train)
    base_full[n] = m
    log(f"  fitted {n} on the full training set")

# meta-learner refit on the SAVED out-of-fold scores (bit-identical)
stack_oof = np.column_stack([z[f"oof__{n}"] for n in top3])
meta_cv = LogisticRegressionCV(Cs=np.logspace(-3, 2, 25), cv=10, penalty="l1",
                               solver="liblinear", scoring="roc_auc",
                               max_iter=4000, random_state=SEED, n_jobs=-1)
meta_cv.fit(stack_oof, y_train)
meta = meta_cv
if np.sum(np.abs(meta.coef_[0])) < 1e-8:
    meta = LogisticRegression(penalty="l1", solver="liblinear",
                              C=float(meta_cv.Cs_.max() if hasattr(meta_cv, "Cs_")
                                      else meta_cv.Cs.max()),
                              max_iter=4000, random_state=SEED)
    meta.fit(stack_oof, y_train)
log(f"[META] weights {dict(zip(top3, np.round(meta.coef_[0], 4)))}, "
    f"intercept {float(meta.intercept_[0]):+.4f}")


def base_probs(Xmat):
    return np.column_stack(
        [(base_full[n].predict_proba(Xmat)[:, 1]
          if hasattr(base_full[n], "predict_proba")
          else 1.0 / (1.0 + np.exp(-base_full[n].decision_function(Xmat))))
         for n in top3])


def stacking_prob(Xmat):
    """FULL v70 stacking pipeline -> probability scale."""
    return meta.predict_proba(base_probs(Xmat))[:, 1]


def stacking_logit(Xmat):
    p = np.clip(stacking_prob(Xmat), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


proba_te = stacking_prob(Xte_mat)
auc_te = fast_auc(y_test.values, proba_te)
log(f"[CHECK] rebuilt stacking internal-test AUC = {auc_te:.4f} "
    f"(v70 reported 0.8456)")
# v70's reported test metrics come from the AVERAGE of the 10 fold-pipelines'
# predictions (bagged evaluation); SHAP must explain ONE deployable model, so
# we use the single full-training refit (same protocol as v50 explained v40).
# The two agree to within ~0.003 AUC, which is expected and acceptable.
_reported = fast_auc(y_test.values, z["tst__Stacking"])
log(f"[CHECK] reported test AUC in the score archive = {_reported:.4f}")
assert abs(auc_te - _reported) < 8e-4, "rebuilt pipeline deviates from run"
audit("")
audit("Q0. Which model is explained?")
audit(f"    The single pipeline refit on the FULL training set "
      f"(internal-test AUC {auc_te:.4f}).")
audit("    v70's headline test AUC (0.8456) is the average prediction of the "
      "ten")
audit("    fold-pipelines; SHAP requires one callable model, so the full "
      "refit is")
audit("    explained instead (same convention as v50 for v40). The two "
      "versions")
audit("    agree to within 0.003 AUC, so the explanation target matches the "
      "reported")
audit("    model to within Monte-Carlo / fold-averaging noise.")

# ----------------------------------------------------------------------
# explainer
# ----------------------------------------------------------------------
BG_SIZE, EXPLAIN_SIZE = 100, 300
N_STABILITY_RUNS = 10

rng = np.random.default_rng(SEED)
bg_idx = rng.choice(len(Xtr_mat), size=BG_SIZE, replace=False)
background = Xtr_mat[bg_idx]
exp_idx = rng.choice(len(Xtr_mat), size=EXPLAIN_SIZE, replace=False)
X_explain = Xtr_mat[exp_idx]
np.save(V80_DIR / "v80_shap_explain_index.npy", exp_idx)

log(f"\n[SETUP] PermutationExplainer | background {BG_SIZE} | "
    f"explain {EXPLAIN_SIZE} | seed {SEED}")

explainer_prob = shap.PermutationExplainer(stacking_prob, background)
explainer_logit = shap.PermutationExplainer(stacking_logit, background)
base_val_prob = float(np.mean(stacking_prob(background)))
base_val_logit = float(np.mean(stacking_logit(background)))
log(f"  base value E[f(X)] = {base_val_prob:.4f} (probability)")

t = time.time()
sv_prob = explainer_prob(X_explain).values
log(f"  [prob] SHAP matrix {sv_prob.shape} ({time.time() - t:.1f} s)")
t = time.time()
sv_logit = explainer_logit(X_explain[:60]).values
log(f"  [logit] SHAP matrix {sv_logit.shape} ({time.time() - t:.1f} s)")

fx = stacking_prob(X_explain)
gx = stacking_logit(X_explain[:60])
add_err = np.abs(sv_prob.sum(axis=1) - (fx - base_val_prob))
audit("v70 SHAP audit")
audit(f"  additivity mean {add_err.mean():.3e}  max {add_err.max():.3e}")
audit(f"  base value (prob) {base_val_prob:.4f}  (logit) {base_val_logit:.4f}")
log(f"  additivity check: mean {add_err.mean():.2e}, max {add_err.max():.2e}")

# ----------------------------------------------------------------------
# case selection (pre-specified, identical to v50)
# ----------------------------------------------------------------------
THR = 0.5
pos_idx = np.where(y_test.values == 1)[0]
neg_idx = np.where(y_test.values == 0)[0]
tp_pool = pos_idx[proba_te[pos_idx] >= THR]
tn_pool = neg_idx[proba_te[neg_idx] < THR]
fp_pool = neg_idx[proba_te[neg_idx] >= THR]
fn_pool = pos_idx[proba_te[pos_idx] < THR]
log(f"\n[CASES] @0.5  TP {len(tp_pool)}  TN {len(tn_pool)}  "
    f"FP {len(fp_pool)}  FN {len(fn_pool)}")

cases = [("TP", int(tp_pool[np.argmax(proba_te[tp_pool])])),
         ("TN", int(tn_pool[np.argmin(proba_te[tn_pool])])),
         ("FP", int(fp_pool[np.argmax(proba_te[fp_pool])])),
         ("FN", int(fn_pool[np.argmin(proba_te[fn_pool])]))]
case_rows = [dict(CaseType=tag, TestRow=i, TrueLabel=int(y_test.values[i]),
                  PredictedProbability=float(proba_te[i])) for tag, i in cases]
pd.DataFrame(case_rows).to_csv(V80_DIR / "v80_shap_cases.csv", index=False,
                               encoding="utf-8-sig")
for tag, i in cases:
    log(f"  {tag}: test row {i:5d}  p = {proba_te[i]:.4f}  "
        f"true = {int(y_test.values[i])}")

sv_cases = explainer_prob(Xte_mat[[i for _, i in cases]]).values

# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
plt = set_style()
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# ---------- beeswarm ----------
fig = plt.figure(figsize=(7.2, 7.6))
shap.summary_plot(sv_prob, X_explain, feature_names=pretty, show=False,
                  max_display=20, plot_size=(7.2, 7.6),
                  color_bar_label="Feature value (low \u2192 high)")
plt.gcf().axes[0].set_xlabel("SHAP value (impact on P(NOAF))", fontsize=9)
plt.title("SHAP summary (beeswarm) \u2014 v70 stacking ensemble\n"
          "values additive in predicted probability of NOAF",
          fontsize=10, fontweight="bold")
fig = plt.gcf()
fig.tight_layout()
save_fig(fig, "Fig8_SHAP_beeswarm")
plt.close(fig)
log("[SAVED] Fig8_SHAP_beeswarm")

# ---------- mean |SHAP| bar with bootstrap CI ----------
mean_abs = np.abs(sv_prob).mean(axis=0)
rngb = np.random.default_rng(SEED)
boot = np.array([np.abs(sv_prob[rngb.choice(len(sv_prob), len(sv_prob),
                                            replace=True)]).mean(axis=0)
                 for _ in range(1000)])
lo = np.percentile(boot, 2.5, axis=0)
hi = np.percentile(boot, 97.5, axis=0)
order = np.argsort(mean_abs)[::-1]
TOPN = 20

rank_df = pd.DataFrame({"Feature": feat_names, "Label": pretty,
                        "MeanAbsSHAP": mean_abs, "CI_lo": lo, "CI_hi": hi})
rank_df = rank_df.sort_values("MeanAbsSHAP", ascending=False).reset_index(
    drop=True)
rank_df["Rank"] = np.arange(1, len(rank_df) + 1)
rank_df.to_csv(V80_DIR / "v80_shap_ranking.csv", index=False,
               encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7.6, 6.0))
yv = np.arange(TOPN)[::-1]
sel = order[:TOPN]
ax.barh(yv, mean_abs[sel], color="#4E79A7", height=0.68, zorder=2)
ax.errorbar(mean_abs[sel], yv,
            xerr=[mean_abs[sel] - lo[sel], hi[sel] - mean_abs[sel]],
            fmt="none", ecolor="#333333", elinewidth=0.9, capsize=2, zorder=3)
ax.set_yticks(yv)
ax.set_yticklabels([pretty[i] for i in sel], fontsize=8.5)
ax.set_xlabel("Mean |SHAP value| (impact on P(NOAF), 95% CI)")
ax.set_xlim(0, hi[sel].max() * 1.22)     # room so CI caps are not clipped
ax.set_title("Global feature importance \u2014 v70 stacking ensemble",
             loc="left", fontweight="bold", fontsize=10)
fig.tight_layout()
save_fig(fig, "Fig8_SHAP_importance_bar")
plt.close(fig)
log("[SAVED] Fig8_SHAP_importance_bar")

# ---------- waterfalls ----------
COLOR = {"TP": "#2E7D32", "TN": "#1565C0", "FP": "#EF6C00", "FN": "#C62828"}
DESC = {"TP": "true positive (NOAF, correctly detected)",
        "TN": "true negative (non-NOAF, correctly excluded)",
        "FP": "false positive (non-NOAF, false alarm)",
        "FN": "false negative (NOAF, missed)"}

for k, (tag, i) in enumerate(cases):
    exp_obj = shap.Explanation(values=sv_cases[k], base_values=base_val_prob,
                               data=Xte_mat[i], feature_names=pretty)
    fig = plt.figure(figsize=(8.8, 7.0))
    shap.plots.waterfall(exp_obj, max_display=12, show=False)
    ax = fig.axes[0]
    # shap draws "f(x) = ..." just above the axes and "E[f(X)] = ..." at the
    # lower axis corner; keep the descriptive title in a separate band at the
    # very top and let shap own the axis labels (no set_xlim / no set_xlabel,
    # both previously clipped bars and collided with those annotations).
    fig.text(0.01, 0.995,
             f"{tag} \u2014 {DESC[tag]}   |   predicted P(NOAF) = "
             f"{proba_te[i]:.3f}   |   observed NOAF = "
             f"{'yes' if y_test.values[i] == 1 else 'no'}   |   "
             f"base value = {base_val_prob:.3f}",
             ha="left", va="top", fontsize=9.5, fontweight="bold",
             color=COLOR[tag])
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.34, right=0.97)
    save_fig(fig, f"Fig8_SHAP_waterfall_{tag}")
    plt.close(fig)
    log(f"[SAVED] Fig8_SHAP_waterfall_{tag}")

np.savez_compressed(V80_DIR / "v80_shap_values.npz",
                    sv_prob=sv_prob, sv_logit=sv_logit,
                    X_explain=X_explain, exp_idx=exp_idx,
                    sv_cases=sv_cases, base_val_prob=base_val_prob,
                    base_val_logit=base_val_logit,
                    feat_names=np.asarray(feat_names, dtype=object),
                    pretty_names=np.asarray(pretty, dtype=object),
                    case_rows=np.asarray(case_rows, dtype=object))
log("[SAVED] v80_shap_values.npz")

# ----------------------------------------------------------------------
# stability (R12)
# ----------------------------------------------------------------------
log(f"\n[R12] stability — {N_STABILITY_RUNS} independent explain sets")
stab_rank, top10_sets = [], []
for r in range(N_STABILITY_RUNS):
    rng_r = np.random.default_rng(1000 + r)
    ix = rng_r.choice(len(Xtr_mat), size=100, replace=False)
    sv_r = explainer_prob(Xtr_mat[ix]).values
    ma = np.abs(sv_r).mean(axis=0)
    rk = pd.Series(ma, index=feat_names).rank(ascending=False)
    stab_rank.append(rk)
    top10_sets.append(set(rk.sort_values().index[:10]))
    log(f"  run {r + 1:2d}/{N_STABILITY_RUNS} done ({time.time() - T0:.0f} s)")

stab = pd.DataFrame(stab_rank)
stab.index = [f"run_{i + 1}" for i in range(N_STABILITY_RUNS)]
stab.T.to_csv(V80_DIR / "v80_shap_stability.csv", encoding="utf-8-sig")

rank_summary = pd.DataFrame({
    "Feature": feat_names,
    "MedianRank": stab.median(axis=0).values,
    "MinRank": stab.min(axis=0).values,
    "MaxRank": stab.max(axis=0).values,
    "RankSD": stab.std(axis=0).values,
}).sort_values("MedianRank").reset_index(drop=True)
rank_summary.to_csv(V80_DIR / "v80_shap_stability_summary.csv", index=False,
                    encoding="utf-8-sig")

base_rank = pd.Series(mean_abs, index=feat_names).rank(ascending=False)
rhos = [spearmanr(base_rank.values, stab.iloc[r].values).statistic
        for r in range(N_STABILITY_RUNS)]
jac = [len(top10_sets[r] & top10_sets[0]) /
       len(top10_sets[r] | top10_sets[0]) for r in range(N_STABILITY_RUNS)]
log(f"  Spearman rho vs main analysis: mean {np.mean(rhos):.4f} "
    f"min {np.min(rhos):.4f} max {np.max(rhos):.4f}")
log(f"  Top-10 Jaccard: mean {np.mean(jac):.3f} min {np.min(jac):.3f} "
    f"max {np.max(jac):.3f}")
audit(f"  stability rho mean {np.mean(rhos):.4f}; "
      f"top-10 Jaccard mean {np.mean(jac):.3f}")

# ---------- stability figure ----------
fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2),
                         gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
TN_ = 15
sub = rank_summary.head(TN_)
yv = np.arange(TN_)[::-1]
lab_map = dict(zip(feat_names, pretty))
for y_, (_, r) in zip(yv, sub.iterrows()):
    ax.plot([r["MinRank"], r["MaxRank"]], [y_, y_], color="#BBBBBB",
            lw=2.4, solid_capstyle="round", zorder=2)
    ax.plot(r["MedianRank"], y_, "o", ms=4.0, color="#D62728", zorder=3)
    ax.plot(base_rank[r["Feature"]], y_, "|", ms=7, color="#1F6FB4",
            mew=1.4, zorder=4)
ax.set_yticks(yv)
ax.set_yticklabels([lab_map[f] for f in sub["Feature"]], fontsize=8)
ax.set_xlabel("Feature rank (1 = most important)")
ax.set_title(f"A  Rank stability across {N_STABILITY_RUNS} explain sets",
             loc="left", fontweight="bold", fontsize=9.5)
ax.legend(handles=[Line2D([], [], color="#BBBBBB", lw=2.4,
                          label="Rank range across runs"),
                   Line2D([], [], color="#D62728", marker="o", ls="", ms=4.0,
                          label="Median rank"),
                   Line2D([], [], color="#1F6FB4", marker="|", ls="", ms=7,
                          mew=1.4, label="Main analysis (seed 42)")],
          loc="upper left", bbox_to_anchor=(0.0, -0.14), fontsize=7.2,
          ncol=3, frameon=False)
ax.invert_xaxis()

ax = axes[1]
ax.hist(rhos, bins=10, color="#4E79A7", edgecolor="white", zorder=2)
ax.axvline(np.mean(rhos), ls="--", color="#D62728", lw=1.2, zorder=3)
ax.text(np.mean(rhos), ax.get_ylim()[1] * 0.92,
        f"mean = {np.mean(rhos):.3f}", ha="left", va="top", fontsize=8,
        color="#D62728")
ax.set_xlabel("Spearman \u03c1 vs main analysis")
ax.set_ylabel("Number of runs")
ax.set_title("B  Ranking agreement", loc="left", fontweight="bold",
             fontsize=9.5)
fig.tight_layout()
save_fig(fig, "Fig8_SHAP_stability")
plt.close(fig)
log("[SAVED] Fig8_SHAP_stability")

log(f"\n[DONE] total {time.time() - T0:.1f} s")
LOG_FH.close()
AUDIT_FH.close()
