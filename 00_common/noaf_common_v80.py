# -*- coding: utf-8 -*-
"""noaf_common_v80.py — thin overlay over noaf_common_v70 that redirects all
outputs to the v80 (fold-internal selection + frozen predictor) directory."""
from noaf_common_v70 import *  # noqa: F401,F403
from noaf_common_v70 import REV_DIR, set_style, save_fig  # noqa: F401

import os as _os

# NOAF_RUN=v90 reruns the whole downstream chain on the fold-internal
# imputation run (v90_primary); default keeps the v80 directory.
_RUN = _os.environ.get("NOAF_RUN", "v80").strip().lower()
V80_DIR = REV_DIR / ("v90_primary" if _RUN in ("v90", "v90_primary") else "v80_foldwise")
FIG_DIR = V80_DIR / "figures"      # shadows noaf_common_v70.FIG_DIR
TAB_DIR = V80_DIR / "tables"
for _d in (V80_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

from noaf_common_v70 import save_fig as _save_fig  # noqa: E402


def save_fig(fig, stem, dirs=(FIG_DIR,)):  # noqa: F811 — bind to v80 dir
    return _save_fig(fig, stem, dirs=dirs)

