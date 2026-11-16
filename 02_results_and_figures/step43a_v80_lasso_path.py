# -*- coding: utf-8 -*-
"""step43a_v80_lasso_path.py — regenerate the LASSO CV-AUC path (Fig 2C input)
on the locked training set (identical procedure to step40 section 2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR, REV_DIR, SEED  # noqa: E402
from noaf_common_v70 import make_preprocessor, split_columns  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402

tr = pd.read_csv(REV_DIR / "v70_leakfree" / "v70_train.csv", encoding="utf-8-sig")
y = tr.pop("NOAF").astype(int)
uni = pd.read_csv(V80_DIR / "v80_univariable_logistic.csv", encoding="utf-8-sig")
uni_vars = list(uni.loc[uni["p"] < 0.05, "var"])
Xu = tr[uni_vars].copy()
num_u, cat_u = split_columns(Xu)
pre = make_preprocessor(num_u, cat_u)
Xmat = pre.fit_transform(Xu)

Cs_grid = np.logspace(-4, 1, 60)
lcv = LogisticRegressionCV(Cs=Cs_grid, cv=10, penalty="l1", solver="liblinear",
                           scoring="roc_auc", max_iter=4000,
                           random_state=SEED, n_jobs=-1)
lcv.fit(Xmat, y)
fs = lcv.scores_[list(lcv.scores_.keys())[0]]
pd.DataFrame(dict(C=lcv.Cs, lambda_=1.0 / lcv.Cs,
                  cv_auc=fs.mean(axis=0),
                  se=fs.std(axis=0) / np.sqrt(fs.shape[0]))
             ).to_csv(V80_DIR / "v80_lasso_path.csv", index=False,
                      encoding="utf-8-sig")
print("[DONE] v80_lasso_path.csv")
