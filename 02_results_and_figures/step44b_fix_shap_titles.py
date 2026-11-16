# -*- coding: utf-8 -*-
"""
step44b: regenerate Fig8_SHAP_beeswarm / Fig8_SHAP_importance_bar from the
saved v80_shap_values.npz with corrected titles ("v80 stacking ensemble",
not "v70"), then step47 must be re-run to rebuild Figure6_SHAP.
No model refitting, no explainer calls - pure re-plot from the npz.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from noaf_common_v80 import V80_DIR, save_fig, SEED  # noqa: E402
from noaf_common_v70 import set_style  # noqa: E402

set_style()  # savefig.dpi=400 etc. — must match step44 rendering conditions

d = np.load(V80_DIR / "v80_shap_values.npz", allow_pickle=True)
sv_prob = d["sv_prob"]
X_explain = d["X_explain"]
feat_names = [str(x) for x in d["feat_names"]]
pretty = [str(x) for x in d["pretty_names"]]

# ---------- beeswarm ----------
fig = plt.figure(figsize=(7.2, 7.6))
shap.summary_plot(sv_prob, X_explain, feature_names=pretty, show=False,
                  max_display=20, plot_size=(7.2, 7.6),
                  color_bar_label="Feature value (low \u2192 high)")
plt.gcf().axes[0].set_xlabel("SHAP value (impact on P(NOAF))", fontsize=9)
plt.title("SHAP summary (beeswarm) \u2014 v80 stacking ensemble\n"
          "values additive in predicted probability of NOAF",
          fontsize=10, fontweight="bold")
fig = plt.gcf()
fig.tight_layout()
save_fig(fig, "Fig8_SHAP_beeswarm")
plt.close(fig)
print("[SAVED] Fig8_SHAP_beeswarm")

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
ax.set_xlim(0, hi[sel].max() * 1.22)
ax.set_title("Global feature importance \u2014 v80 stacking ensemble",
             loc="left", fontweight="bold", fontsize=10)
fig.tight_layout()
save_fig(fig, "Fig8_SHAP_importance_bar")
plt.close(fig)
print("[SAVED] Fig8_SHAP_importance_bar")
print("[DONE] re-run step47 to rebuild Figure6_SHAP")
