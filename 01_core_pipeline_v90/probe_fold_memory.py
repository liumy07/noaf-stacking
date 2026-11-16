# -*- coding: utf-8 -*-
"""probe_fold_memory.py — run ONE fold's phase A inline and log where the
private memory goes, to size a safe worker pool for the parallel run."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time, threading
from pathlib import Path
import numpy as np
import pandas as pd

import ctypes
import ctypes.wintypes as wt


class PMC(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def private_mb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(),
        ctypes.byref(pmc), pmc.cb)
    return pmc.PagefileUsage / 1e6


sys.path.insert(0, str(Path(__file__).resolve().parent))
PEAK = {"mb": 0.0, "stage": "init"}


def monitor():
    while not STOP["t"]:
        mb = private_mb()
        if mb > PEAK["mb"]:
            PEAK["mb"] = mb
            PEAK["stage"] = STAGE["s"]
        time.sleep(1)


STOP = {"t": False}
STAGE = {"s": "loading"}

t = threading.Thread(target=monitor, daemon=True)
t.start()


def mark(s):
    STAGE["s"] = s
    mb = private_mb()
    print(f"[{time.strftime('%H:%M:%S')}] {s:35s} private {mb:8.0f} MB "
          f"(peak so far {PEAK['mb']:.0f} MB)", flush=True)


mark("start")
from noaf_common_v70 import (REV_DIR, VARS39, load_raw_cohorts, split_indices,
                             make_preprocessor, split_columns)
mark("imports done")
X_m, y_m, X_e, y_e = load_raw_cohorts()
itrain, itest = split_indices(y_m)
X_train_raw = X_m.iloc[itrain].reset_index(drop=True)
y_train = y_m.iloc[itrain].reset_index(drop=True)
mark("raw loaded")

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
from noaf_common_v70 import BIN01, BIN_STR

skf_1 = X_train_raw.iloc[:21661]
mark("fold-1 train slice")

t0 = time.time()
est = RandomForestRegressor(n_estimators=100, max_features="sqrt",
                            min_samples_leaf=3, n_jobs=1, random_state=42)
imp = IterativeImputer(estimator=est, max_iter=10, tol=1e-3,
                       initial_strategy="median", imputation_order="ascending",
                       skip_complete=True, random_state=42,
                       sample_posterior=False)
mark("imputer built")
imp.fit(skf_1)
mark(f"imputer FIT done ({time.time()-t0:.0f} s)")

Z = imp.transform(skf_1)
mark("imputer transform train done")
del Z

# a couple of RF fits to see steady-state
for k in range(3):
    e2 = RandomForestRegressor(n_estimators=100, max_features="sqrt",
                               min_samples_leaf=3, n_jobs=1, random_state=42)
    e2.fit(skf_1.iloc[:, 1:], skf_1.iloc[:, 0])
    del e2
    mark(f"bare RF fit {k+1} done")
gc_mark = f"PEAK: {PEAK['mb']:.0f} MB at stage: {PEAK['stage']}"
print("=" * 60)
print(gc_mark, flush=True)
STOP["t"] = True
