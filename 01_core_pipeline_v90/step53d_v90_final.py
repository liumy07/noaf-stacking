# -*- coding: utf-8 -*-
"""
step53d_v90_final.py — v90 sensitivity analysis, final production version.

Same methodology as step53/step53b/step53c (imputer refitted inside every CV
fold on that fold's training rows; locked selection + frozen predictor
identical to v80).

WHY THE IMPUTER IS RE-IMPLEMENTED HERE
--------------------------------------
sklearn's IterativeImputer retains EVERY fitted forest in
`imputation_sequence_`: 26 incomplete variables x 10 iterations = 260 forests,
which grows a fold's process to ~15 GB.  A single serial fold tolerates that;
10 concurrent folds demand ~150 GB and exhaust the commit limit (this machine:
77-99 GB).  The explicit loop below is the canonical missForest algorithm
(Stekhoven & Buhlmann 2012) with the same settings and discards each forest
after use.  Verified numerically identical to sklearn's implementation on a
4,000-row subsample: max |difference| = 1.7e-13 (validate_missforest.py).

    PHASE A (parallel, 10 workers, ~1-2 GB each): per fold — missForest loop
        on fold-train rows only, impute fold-train + fold-val, re-run the
        univariable screen + LASSO 1-SE inside the fold, return the imputed
        frames + selected variables.
    PHASE B (serial, in the parent): nine learners per fold with the EXACT
        v80 code path, then the frozen predictor, top-3, meta-learner.

Outputs: v90_scores.npz, v90_results.csv, v90_selection_audit.csv,
v90_fold_audit.csv, v90_run_log.txt in output/rev_noaf/v90_fold_impute/
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import gc
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

N_WORKERS = 10
N_JOBS_IMPUTER = 2      # threads inside each worker's forests
MISSFOREST_ITER = 10
RF_TREES_IMPUTE = 100
SEED = 42


# =====================================================================
# missForest, explicit and bounded in memory
# =====================================================================
def missforest_fit(X_df, n_iter=MISSFOREST_ITER, n_trees=RF_TREES_IMPUTE,
                   seed=SEED, n_jobs=N_JOBS_IMPUTER, tol=1e-3):
    """Canonical missForest: median initialisation, variables visited in
    ascending order of missingness, each visited variable regressed on all
    others by a random forest fitted on its OBSERVED rows, repeated for
    `n_iter` passes or until the maximum change falls below `tol`.

    Returns (imputed ndarray, last-pass forests, training medians, order).
    Only the last pass's forests are kept, so a fold holds ~26 forests instead
    of sklearn's 260 -> memory stays ~1-2 GB per worker.
    """
    from sklearn.ensemble import RandomForestRegressor

    Xv = X_df.to_numpy(dtype=float, copy=True)
    obs = ~np.isnan(Xv)
    miss_count = (~obs).sum(axis=0)
    med = np.nanmedian(Xv, axis=0)
    r, c = np.where(~obs)
    Xv[r, c] = med[c]
    order = [j for j in np.argsort(miss_count, kind="mergesort")
             if miss_count[j] > 0]

    models = {}
    for it in range(n_iter):
        prev = Xv.copy()
        for j in order:
            o = obs[:, j]
            m = ~o
            if m.sum() == 0 or o.sum() < 10:
                continue
            Xo = np.delete(Xv, j, axis=1)
            rf = RandomForestRegressor(n_estimators=n_trees,
                                       max_features="sqrt",
                                       min_samples_leaf=3,
                                       n_jobs=n_jobs, random_state=seed)
            rf.fit(Xo[o], Xv[o, j])
            Xv[m, j] = rf.predict(Xo[m])
            models[j] = rf
        gc.collect()
        if np.max(np.abs(Xv - prev)) < tol:
            break
    return Xv, models, med, order


def missforest_transform(X_df, models, med, order):
    """Apply a fitted missForest to new rows (one pass, training medians)."""
    Xv = X_df.to_numpy(dtype=float, copy=True)
    obs = ~np.isnan(Xv)
    r, c = np.where(~obs)
    Xv[r, c] = np.take(med, c)
    for j in order:
        m = ~obs[:, j]
        if m.sum() == 0 or j not in models:
            continue
        Xo = np.delete(Xv, j, axis=1)
        Xv[m, j] = models[j].predict(Xo[m])
    return pd.DataFrame(Xv, index=X_df.index, columns=X_df.columns)


# =====================================================================
# PHASE A worker — imputation + fold-internal selection (light imports)
# =====================================================================
def run_fold_phase_a(fold: int, Xtr_raw, ytr, Xva_raw):
    import statsmodels.api as sm
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from noaf_common_v70 import (VARS39, BIN01, BIN_STR, make_preprocessor,
                                 split_columns)

    t0 = time.time()

    # ---- fold-internal imputation (fit on fold-train rows ONLY) --------
    t_imp = time.time()
    Xt_arr, models, med, order = missforest_fit(Xtr_raw)
    Xt = pd.DataFrame(Xt_arr, index=Xtr_raw.index, columns=Xtr_raw.columns)
    Xv = missforest_transform(Xva_raw, models, med, order)

    for Z in (Xt, Xv):                      # binaries back to 0/1
        for c in BIN01 + list(BIN_STR):
            if c in Z:
                Z[c] = Z[c].round().clip(0, 1).astype(int)
    del models, Xt_arr
    gc.collect()
    t_imp = time.time() - t_imp

    # ---- fold-internal univariable screen -------------------------------
    rows = []
    for c in VARS39:
        x = Xt[c].astype(float).values
        if c == "output_urine_first":
            x = x / 100.0
        res = sm.Logit(ytr.values, sm.add_constant(x)).fit(disp=0)
        rows.append((c, float(res.pvalues[1])))
    uni_f = [c for c, p in rows if p < 0.05]

    # ---- fold-internal LASSO 1-SE ----------------------------------------
    X_u = Xt[uni_f].copy()
    num_u, cat_u = split_columns(X_u)
    pre = make_preprocessor(num_u, cat_u)
    Xmat = pre.fit_transform(X_u)
    feat_names = list(pre.get_feature_names_out())
    expanded = {}
    for tok in feat_names:
        if tok.startswith("num__"):
            expanded[tok] = tok[len("num__"):]
        elif tok.startswith("cat__"):
            body = tok[len("cat__"):]
            for cat in cat_u:
                if body.startswith(cat + "_"):
                    expanded[tok] = cat
                    break
    Cs_grid = np.logspace(-4, 1, 60)
    lcv = LogisticRegressionCV(Cs=Cs_grid, cv=10, penalty="l1",
                               solver="liblinear", scoring="roc_auc",
                               max_iter=4000, random_state=SEED, n_jobs=1)
    lcv.fit(Xmat, ytr)
    fs = lcv.scores_[list(lcv.scores_.keys())[0]]
    mean_s, se_s = fs.mean(axis=0), fs.std(axis=0) / np.sqrt(fs.shape[0])
    best = int(np.argmax(mean_s))
    thr = mean_s[best] - se_s[best]
    C_1se = float(lcv.Cs[int(np.where(mean_s >= thr)[0].min())])
    las = LogisticRegression(penalty="l1", solver="liblinear", C=C_1se,
                             max_iter=4000, random_state=SEED)
    las.fit(Xmat, ytr)
    co = pd.Series(las.coef_[0], index=feat_names)
    sel_f = sorted({expanded.get(nm, nm) for nm in co[co.abs() > 1e-6].index})
    if len(sel_f) >= len(uni_f):
        for C_try in np.sort(Cs_grid)[np.sort(Cs_grid) < C_1se][::-1]:
            lt = LogisticRegression(penalty="l1", solver="liblinear",
                                    C=float(C_try), max_iter=4000,
                                    random_state=SEED)
            lt.fit(Xmat, ytr)
            cs = pd.Series(lt.coef_[0], index=feat_names)
            nz = sorted({expanded.get(nm, nm)
                         for nm in cs[cs.abs() > 1e-6].index})
            if len(nz) < len(uni_f):
                C_1se, sel_f = float(C_try), nz
                break
    if not sel_f:
        sel_f = sorted(uni_f)
    t_sel = time.time() - t0 - t_imp

    audit = dict(fold=fold, imputer_s=round(t_imp, 1),
                 selection_s=round(t_sel, 1), n_uni=len(uni_f),
                 n_lasso=len(sel_f), C_1se=C_1se, vars=";".join(sel_f))
    return fold, Xt[sel_f], Xv[sel_f], audit


# =====================================================================
# parent
# =====================================================================
def main():
    from noaf_common_v70 import (REV_DIR, N_FOLDS, VARS39, MODEL_NAMES,
                                 DISPLAY, make_preprocessor, split_columns,
                                 base_models, get_scores, sigmoid,
                                 load_raw_cohorts, split_indices,
                                 fast_auc, average_precision, brier,
                                 calib_slope_intercept)
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    import statsmodels.api as sm

    V90_DIR = REV_DIR / "v90_fold_impute"
    V90_DIR.mkdir(parents=True, exist_ok=True)
    T0 = time.time()
    LOG = open(V90_DIR / "v90_run_log.txt", "w", encoding="utf-8")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        LOG.write(msg + "\n")
        LOG.flush()

    log("=" * 78)
    log(f"v90 — fold-internal imputation sensitivity (two-phase parallel: "
        f"phase A {N_WORKERS} workers, phase B serial)")
    log("=" * 78)

    # 1. raw cohorts + same split as v70/v80 (asserted)
    log("\n[1] Loading raw (pre-imputation) cohorts ...")
    X_m, y_m, X_e, y_e = load_raw_cohorts()
    itrain, itest = split_indices(y_m)
    X_train_raw = X_m.iloc[itrain].reset_index(drop=True)
    y_train = y_m.iloc[itrain].reset_index(drop=True)

    tr70 = pd.read_csv(REV_DIR / "v70_leakfree" / "v70_train.csv",
                       encoding="utf-8-sig")
    y70 = tr70.pop("NOAF").astype(int)
    assert len(y70) == len(y_train)
    assert (y70.values == y_train.values).all()
    te = pd.read_csv(REV_DIR / "v70_leakfree" / "v70_test.csv",
                     encoding="utf-8-sig")
    ei = pd.read_csv(REV_DIR / "v70_leakfree" / "v70_eicu.csv",
                     encoding="utf-8-sig")
    y_test = te.pop("NOAF").astype(int)
    y_eicu = ei.pop("NOAF").astype(int)
    X_train_imp, X_test_imp, X_eicu_imp = tr70, te, ei
    assert list(X_train_imp.columns) == VARS39
    log(f"  raw train {len(y_train):,} (matches v70) | test {len(y_test):,} "
        f"| eICU {len(y_eicu):,}")

    # 2. locked final selection (identical to v80)
    log("\n[LOCKED] univariable + LASSO 1-SE on the full imputed training set ...")
    rows = []
    for c in VARS39:
        x = X_train_imp[c].astype(float).values
        if c == "output_urine_first":
            x = x / 100.0
        res = sm.Logit(y_train.values, sm.add_constant(x)).fit(disp=0)
        rows.append((c, float(res.pvalues[1])))
    uni_vars = [c for c, p in rows if p < 0.05]
    X_u = X_train_imp[uni_vars].copy()
    num_u, cat_u = split_columns(X_u)
    pre = make_preprocessor(num_u, cat_u)
    Xmat = pre.fit_transform(X_u)
    feat_names = list(pre.get_feature_names_out())
    expanded = {}
    for tok in feat_names:
        if tok.startswith("num__"):
            expanded[tok] = tok[len("num__"):]
        elif tok.startswith("cat__"):
            body = tok[len("cat__"):]
            for cat in cat_u:
                if body.startswith(cat + "_"):
                    expanded[tok] = cat
                    break
    Cs_grid = np.logspace(-4, 1, 60)
    lcv = LogisticRegressionCV(Cs=Cs_grid, cv=10, penalty="l1",
                               solver="liblinear", scoring="roc_auc",
                               max_iter=4000, random_state=SEED, n_jobs=1)
    lcv.fit(Xmat, y_train)
    fs = lcv.scores_[list(lcv.scores_.keys())[0]]
    mean_s, se_s = fs.mean(axis=0), fs.std(axis=0) / np.sqrt(fs.shape[0])
    best = int(np.argmax(mean_s))
    thr = mean_s[best] - se_s[best]
    C_final = float(lcv.Cs[int(np.where(mean_s >= thr)[0].min())])
    las = LogisticRegression(penalty="l1", solver="liblinear", C=C_final,
                             max_iter=4000, random_state=SEED)
    las.fit(Xmat, y_train)
    co = pd.Series(las.coef_[0], index=feat_names)
    sel_vars = sorted({expanded.get(nm, nm)
                       for nm in co[co.abs() > 1e-6].index})
    if len(sel_vars) >= len(uni_vars):
        for C_try in np.sort(Cs_grid)[np.sort(Cs_grid) < C_final][::-1]:
            lt = LogisticRegression(penalty="l1", solver="liblinear",
                                    C=float(C_try), max_iter=4000,
                                    random_state=SEED)
            lt.fit(Xmat, y_train)
            cs = pd.Series(lt.coef_[0], index=feat_names)
            nz = sorted({expanded.get(nm, nm)
                         for nm in cs[cs.abs() > 1e-6].index})
            if len(nz) < len(uni_vars):
                C_final, sel_vars = float(C_try), nz
                break
    if not sel_vars:
        sel_vars = sorted(uni_vars)
    log(f"  univariable {len(uni_vars)}/39; LASSO 1-SE (C={C_final:.6g}) "
        f"retained {len(sel_vars)}: {sel_vars}")

    X_tr_sel = X_train_imp[sel_vars].copy()
    X_te_sel = X_test_imp[sel_vars].copy()
    X_ei_sel = X_eicu_imp[sel_vars].copy()

    # 3. PHASE A — 10 folds in parallel (imputation + selection)
    log(f"\n[PHASE A] imputer REFIT inside every fold "
        f"({N_WORKERS} concurrent light workers)")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(X_train_raw, y_train))
    fold_data = {}
    audits = []
    CACHE = V90_DIR / "phase_a_cache"
    CACHE.mkdir(exist_ok=True)
    have_cache = all((CACHE / f"fold{k}.pkl").exists()
                     for k in range(1, N_FOLDS + 1))
    if have_cache:
        import pickle
        for k in range(1, N_FOLDS + 1):
            with open(CACHE / f"fold{k}.pkl", "rb") as fh:
                d = pickle.load(fh)
            fold_data[k] = (d["itr"], d["iva"], d["Xf"], d["Xv"])
            audits.append(d["audit"])
        log("  [PHASE A] loaded from cache — imputation not repeated")
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futs = {}
            for fold, (itr, iva) in enumerate(splits, 1):
                futs[ex.submit(run_fold_phase_a, fold,
                               X_train_raw.iloc[itr], y_train.iloc[itr],
                               X_train_raw.iloc[iva])] = (itr, iva)
            for fut in as_completed(futs):
                itr, iva = futs[fut]
                fold, Xf_sel, Xv_sel, audit = fut.result()
                fold_data[fold] = (itr, iva, Xf_sel, Xv_sel)
                # cache to disk so a downstream failure never repeats phase A
                import pickle
                with open(CACHE / f"fold{fold}.pkl", "wb") as fh:
                    pickle.dump(dict(itr=itr, iva=iva, Xf=Xf_sel, Xv=Xv_sel,
                                     audit=audit), fh)
                audits.append(audit)
                log(f"  fold {fold:2d} impute {audit['imputer_s']:6.1f}s  "
                    f"sel {audit['selection_s']:5.1f}s  "
                    f"uni {audit['n_uni']:2d} -> lasso {audit['n_lasso']:2d} "
                    f"(C={audit['C_1se']:.4g})")
        log(f"  [PHASE A] done at {time.time() - T0:.0f} s")
    audits.sort(key=lambda r: r["fold"])
    pd.DataFrame(audits).to_csv(V90_DIR / "v90_fold_audit.csv",
                                index=False, encoding="utf-8-sig")

    # 4. PHASE B — nine learners per fold, serial, exact v80 code path
    log(f"\n[PHASE B] 9 learners x {N_FOLDS} folds (serial, v80 settings)")
    oof = {n_: np.zeros(len(y_train)) for n_ in MODEL_NAMES}
    for fold in range(1, N_FOLDS + 1):
        itr, iva, Xf_sel, Xv_sel = fold_data[fold]
        ytr = y_train.iloc[itr]
        num_f, cat_f = split_columns(Xf_sel)
        pre_f = make_preprocessor(num_f, cat_f)
        tb = time.time()
        for name, clf in base_models().items():
            pipe = Pipeline([("pre", pre_f), ("clf", clone(clf))])
            pipe.fit(Xf_sel, ytr)
            oof[name][iva] = get_scores(pipe, Xv_sel)
        log(f"  fold {fold:2d} learners done ({time.time()-tb:5.1f} s)")

    # 5. frozen predictor — identical to v80 (parent, untouched settings)
    log("\n[FREEZE] refitting all nine learners on the full imputed "
        "training set ...")
    models = base_models()
    num_sel, cat_sel = split_columns(X_tr_sel)
    preprocessor = make_preprocessor(num_sel, cat_sel)
    frozen = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", clone(clf))])
        pipe.fit(X_tr_sel, y_train)
        frozen[name] = pipe
    ins = {n_: get_scores(frozen[n_], X_tr_sel) for n_ in MODEL_NAMES}
    tst = {n_: get_scores(frozen[n_], X_te_sel) for n_ in MODEL_NAMES}
    ext = {n_: get_scores(frozen[n_], X_ei_sel) for n_ in MODEL_NAMES}
    for n_ in MODEL_NAMES:
        if not hasattr(models[n_], "predict_proba"):
            ins[n_] = sigmoid(ins[n_]); tst[n_] = sigmoid(tst[n_])
            ext[n_] = sigmoid(ext[n_])

    # 6. top-3 by v90 OOF AUC (MO05 tie rule)
    rows = []
    for n_ in MODEL_NAMES:
        rows.append(dict(
            Model=n_, Display=DISPLAY[n_],
            AUC_train_OOF=fast_auc(y_train.values, oof[n_]),
            AP_train_OOF=average_precision(y_train.values, oof[n_]),
            AUC_internal_test=fast_auc(y_test.values, tst[n_]),
            AUC_eICU=fast_auc(y_eicu.values, ext[n_]),
            AUC_train_OOF_4dp=round(fast_auc(y_train.values, oof[n_]), 4)))
    sel_df = pd.DataFrame(rows)
    order = {n_: i for i, n_ in enumerate(MODEL_NAMES)}
    sel_df["_o"] = sel_df["Model"].map(order)
    sel_df = sel_df.sort_values(["AUC_train_OOF", "AP_train_OOF", "_o"],
                                ascending=[False, False, True]).reset_index(drop=True)
    sel_df["Rank_OOF"] = sel_df.index + 1
    sel_df.drop(columns="_o").to_csv(V90_DIR / "v90_selection_audit.csv",
                                     index=False, encoding="utf-8-sig")
    log(f"  {'Model':10s} {'OOF':>9s} {'test':>9s} {'eICU':>9s}   rank")
    for _, r in sel_df.iterrows():
        log(f"  {r['Model']:10s} {r['AUC_train_OOF']:9.6f} "
            f"{r['AUC_internal_test']:9.6f} {r['AUC_eICU']:9.6f}   "
            f"{r['Rank_OOF']}")
    top3 = list(sel_df.head(3)["Model"])
    log(f"  Top-3 locked: {top3}")

    # 7. LASSO meta-learner on the v90 OOF matrix
    stack_oof = np.column_stack([oof[n_] for n_ in top3])
    stack_te = np.column_stack([tst[n_] for n_ in top3])
    stack_ei = np.column_stack([ext[n_] for n_ in top3])
    stack_in = np.column_stack([ins[n_] for n_ in top3])
    meta_cv = LogisticRegressionCV(Cs=np.logspace(-3, 2, 25), cv=10,
                                   penalty="l1", solver="liblinear",
                                   scoring="roc_auc", max_iter=4000,
                                   random_state=SEED, n_jobs=-1)
    meta_cv.fit(stack_oof, y_train)
    meta = meta_cv
    if np.sum(np.abs(meta.coef_[0])) < 1e-8:
        meta = LogisticRegression(penalty="l1", solver="liblinear",
                                  C=float(meta_cv.Cs.max()), max_iter=4000,
                                  random_state=SEED).fit(stack_oof, y_train)
    oof["Stacking"] = meta.predict_proba(stack_oof)[:, 1]
    tst["Stacking"] = meta.predict_proba(stack_te)[:, 1]
    ext["Stacking"] = meta.predict_proba(stack_ei)[:, 1]
    ins["Stacking"] = meta.predict_proba(stack_in)[:, 1]
    log("\n[META] LASSO meta-learner on v90 OOF")
    for nm, w in zip(top3, meta.coef_[0]):
        log(f"  {nm:10s} {w:+.4f}")
    log(f"  intercept {float(meta.intercept_[0]):+.4f}  "
        f"C = {float(meta.C_[0]):.6g}")

    # 8. headline metrics + cache
    rows = []
    for n_ in MODEL_NAMES + ["Stacking"]:
        for cohort, yy, pp in (("train OOF", y_train.values, oof[n_]),
                               ("internal test", y_test.values, tst[n_]),
                               ("eICU external", y_eicu.values, ext[n_])):
            a, b, _, _ = calib_slope_intercept(yy, pp)
            rows.append(dict(Model=n_, cohort=cohort,
                             AUC=fast_auc(yy, pp),
                             AP=average_precision(yy, pp),
                             Brier=brier(yy, pp),
                             calib_intercept=a, calib_slope=b,
                             prev_obs=float(np.mean(yy)),
                             prev_pred=float(np.mean(pp))))
    pd.DataFrame(rows).to_csv(V90_DIR / "v90_results.csv", index=False,
                              encoding="utf-8-sig")
    log("\n[RESULTS] v90 (frozen predictor, internal test / eICU)")
    res = pd.DataFrame(rows)
    for n_ in ["Stacking"] + top3:
        for cohort in ("train OOF", "internal test", "eICU external"):
            r = res[(res.Model == n_) & (res.cohort == cohort)].iloc[0]
            log(f"  {n_:10s} {cohort:15s} AUC {r.AUC:.6f}  AP {r.AP:.4f}  "
                f"Brier {r.Brier:.4f}  slope {r.calib_slope:.3f}")

    payload = {"y_train": y_train.values.astype(int),
               "y_test": y_test.values.astype(int),
               "y_eicu": y_eicu.values.astype(int),
               "meta_weights": np.asarray(meta.coef_[0], dtype=float),
               "meta_intercept": float(meta.intercept_[0]),
               "meta_C": float(meta.C_[0]),
               "top3": np.asarray(top3, dtype=object),
               "selected_vars": np.asarray(sel_vars, dtype=object),
               "n_uni_full": len(uni_vars)}
    for n_ in MODEL_NAMES + ["Stacking"]:
        payload[f"oof__{n_}"] = oof[n_]
        payload[f"tst__{n_}"] = tst[n_]
        payload[f"ext__{n_}"] = ext[n_]
        payload[f"ins__{n_}"] = ins[n_]
    np.savez_compressed(V90_DIR / "v90_scores.npz", **payload)
    log(f"\n[SAVED] v90_scores.npz   [DONE] {time.time() - T0:.1f} s")
    LOG.close()


if __name__ == "__main__":
    main()
