# -*- coding: utf-8 -*-
"""
validate_missforest.py — check that the explicit missForest loop (which
discards each forest after use) reproduces sklearn's IterativeImputer
(which retains all 260 fitted forests and therefore leaks ~15 GB per fold).

Run on a 4,000-row subsample with 2 iterations; both implementations use
identical settings (median init, ascending order, RF 100 trees,
max_features=sqrt, min_samples_leaf=3, tol=1e-3).
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v70 import (VARS39, load_raw_cohorts, split_indices)


def missforest_explicit(X_df, n_iter=10, n_trees=100, seed=42, n_jobs=1,
                        tol=1e-3):
    X = X_df.to_numpy(dtype=float, copy=True)
    obs = ~np.isnan(X)
    miss = (~obs).sum(axis=0)
    med = np.nanmedian(X, axis=0)
    r, c = np.where(~obs)
    X[r, c] = med[c]
    order = [j for j in np.argsort(miss, kind="mergesort") if miss[j] > 0]

    from sklearn.ensemble import RandomForestRegressor
    for it in range(n_iter):
        prev = X.copy()
        for j in order:
            o = obs[:, j]
            m = ~o
            if m.sum() == 0 or o.sum() < 10:
                continue
            Xo = np.delete(X, j, axis=1)
            rf = RandomForestRegressor(n_estimators=n_trees,
                                       max_features="sqrt",
                                       min_samples_leaf=3,
                                       n_jobs=n_jobs, random_state=seed)
            rf.fit(Xo[o], X[o, j])
            X[m, j] = rf.predict(Xo[m])
            del rf
        gc.collect()
        delta = np.max(np.abs(X - prev))
        print(f"  iteration {it+1}: max change {delta:.2e}", flush=True)
        if delta < tol:
            break
    return pd.DataFrame(X, index=X_df.index, columns=X_df.columns)


def main():
    X_m, y_m, X_e, y_e = load_raw_cohorts()
    itrain, _ = split_indices(y_m)
    sub = X_m.iloc[itrain][:4000].reset_index(drop=True)
    print(f"subsample {sub.shape}; missing cells {int(sub.isna().sum().sum())}")

    # --- reference: sklearn IterativeImputer -----------------------------
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import IterativeImputer
    t0 = time.time()
    ref = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, max_features="sqrt",
                                        min_samples_leaf=3, n_jobs=-1,
                                        random_state=42),
        max_iter=2, tol=1e-3, initial_strategy="median",
        imputation_order="ascending", skip_complete=True,
        random_state=42, sample_posterior=False).fit_transform(sub)
    t_ref = time.time() - t0
    print(f"sklearn reference done in {t_ref:.1f} s")

    # --- our explicit loop ------------------------------------------------
    t0 = time.time()
    mine = missforest_explicit(sub, n_iter=2, n_jobs=-1)
    t_mine = time.time() - t0
    print(f"explicit loop done in {t_mine:.1f} s")

    a = np.asarray(ref, dtype=float)
    b = np.asarray(mine, dtype=float)
    diff = np.abs(a - b)
    print("=" * 60)
    print(f"max |diff| = {diff.max():.3e}")
    print(f"mean |diff| = {diff.mean():.3e}")
    print(f"corr = {np.corrcoef(a.ravel(), b.ravel())[0,1]:.8f}")
    print(f"relative mean diff = {diff.mean()/np.abs(a).mean():.2e}")


if __name__ == "__main__":
    main()
