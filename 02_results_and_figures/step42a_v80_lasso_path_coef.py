# -*- coding: utf-8 -*-
"""step30a_lasso_path_coef.py — recompute the LASSO coefficient path on the
v70 training set (same preprocessing / grid as step21) for Figure 2B."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR, REV_DIR, VARS39, split_columns, make_preprocessor  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402

SEED = 42
tr = pd.read_csv(REV_DIR / "v70_leakfree" / "v70_train.csv", encoding="utf-8-sig")
y = tr.pop("NOAF").astype(int)
X = tr[VARS39]

uni = pd.read_csv(V80_DIR / "v80_univariable_logistic.csv", encoding="utf-8-sig")
selected_uni = list(uni.loc[uni["p"] < 0.05, "var"])
Xu = X[selected_uni].copy()
num_u, cat_u = split_columns(Xu)
pre = make_preprocessor(num_u, cat_u)
Xt = pre.fit_transform(Xu)
feat_names = list(pre.get_feature_names_out())
expanded_to_orig = {}
for tok in feat_names:
    if tok.startswith("num__"):
        expanded_to_orig[tok] = tok[len("num__"):]
    elif tok.startswith("cat__"):
        body = tok[len("cat__"):]
        for cat in cat_u:
            if body.startswith(cat + "_"):
                expanded_to_orig[tok] = cat
                break

Cs = np.logspace(-4, 1, 60)
rows = []
for i, C in enumerate(Cs, 1):
    lr = LogisticRegression(penalty="l1", solver="liblinear", C=float(C),
                            max_iter=4000, random_state=SEED)
    lr.fit(Xt, y)
    coefs = pd.Series(lr.coef_[0], index=feat_names)
    mag = coefs.abs().groupby(expanded_to_orig).max()
    rows.append(dict(step=i, C=float(C), lambda_=1.0 / float(C), **mag.to_dict()))
    print(f"  {i:2d}/60  C={C:.4g}  nonzero={int((coefs.abs() > 1e-6).sum())}",
          flush=True)

df = pd.DataFrame(rows)
df.to_csv(V80_DIR / "v80_lasso_path_coef.csv", index=False, encoding="utf-8-sig")
print("[DONE]")
