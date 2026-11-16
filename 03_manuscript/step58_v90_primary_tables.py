# -*- coding: utf-8 -*-
"""step58_v90_primary_tables.py

Rewrite Table 2 (nine models) and the appended Supplementary Table S4
(paired DeLong) in the manuscript with the v90 (fold-internal imputation)
numbers, preserving the existing cell formatting.

Run with NOAF_RUN=v90 so the v90_primary CSVs are used.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))
from noaf_common_v80 import V80_DIR  # noqa: E402

DOCX = Path(r"D:/BJND-4002-11项目/BJND-4002-11/NOAF_stacking_manuscript_EndNote_revised_v90.docx")
DASH = "\u2013"

ROW_ORDER = ["Logistic", "Ridge", "DT", "KNN", "SVM", "MLP", "RF",
             "LightGBM", "XGBoost", "Stacking"]
DISP = {"MLP": "Multilayer perceptron", "XGBoost": "XGBoost",
        "RF": "Random forest", "LightGBM": "LightGBM",
        "Logistic": "Logistic regression", "SVM": "Support vector machine",
        "Ridge": "Ridge regression", "DT": "Decision tree",
        "KNN": "K-nearest neighbours", "Stacking": "Stacking ensemble"}
COH = {"test": "MIMIC-IV internal test (n = 10,316)",
       "eICU": "eICU-CRD external (n = 32,067)"}


def set_cell(cell, text):
    """Replace the cell text while keeping the first run's formatting."""
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        p.add_run(text)


def fmt(v, lo, hi):
    return f"{v:.3f} ({lo:.3f}{DASH}{hi:.3f})"


def main():
    auc = pd.read_csv(V80_DIR / "v80_nine_models_metrics.csv", encoding="utf-8-sig")
    ap = pd.read_csv(V80_DIR / "v80_ci_ap.csv", encoding="utf-8-sig")
    dl = pd.read_csv(V80_DIR / "v80_delong_paired.csv", encoding="utf-8-sig")

    doc = Document(DOCX)
    t2 = doc.tables[1]
    assert len(t2.rows) == 12, f"unexpected Table 2 shape {len(t2.rows)}"

    for i, m in enumerate(ROW_ORDER, start=2):
        row = t2.rows[i]
        assert row.cells[0].text.strip() == DISP[m], (i, row.cells[0].text)
        for j, coh in enumerate(["train", "test", "eICU"]):
            a = auc[(auc.Model == m) & (auc.cohort == coh)].iloc[0]
            b = ap[(ap.Model == m) & (ap.cohort == coh)].iloc[0]
            set_cell(row.cells[1 + 2 * j], fmt(a.AUC, a.CI_lo, a.CI_hi))
            set_cell(row.cells[2 + 2 * j], fmt(b.AP, b.lo, b.hi))

    # ---- Supplementary Table S4 (appended at the end of the document) ----
    s4 = doc.tables[2]
    assert len(s4.rows) == 1 + len(dl), (len(s4.rows), len(dl))
    name2key = {v: k for k, v in DISP.items() if k != "Stacking"}
    for i in range(1, len(s4.rows)):
        cells = s4.rows[i].cells
        label = cells[0].text.strip()
        cohort = "eICU" if "eICU" in label else "test"
        base = next(k for v, k in name2key.items() if v in label)
        r = dl[(dl.base == base) & (dl.cohort == cohort)]
        assert len(r) == 1, (label, base, cohort, len(r))
        r = r.iloc[0]
        set_cell(cells[1], f"{r.delta:+.4f}")
        set_cell(cells[2], f"{r.lo:+.4f} to {r.hi:+.4f}")
        set_cell(cells[3], "P < 0.001" if r.p < 1e-3 else f"{r.p:.3f}")

    bak = DOCX.with_name(DOCX.stem + "_pre_tables_v90.docx")
    if not bak.exists():
        shutil.copy2(DOCX, bak)
    doc.save(DOCX)
    print("[SAVED]", DOCX.name)

    for r in t2.rows[11].cells:
        pass
    print("Table 2 stacking row:",
          " | ".join(c.text.strip() for c in t2.rows[11].cells))
    print("S4 row (XGBoost, eICU):")
    for r in s4.rows[1:]:
        if "XGBoost" in r.cells[0].text and "eICU" in r.cells[0].text:
            print("   ", " | ".join(c.text.strip() for c in r.cells))


if __name__ == "__main__":
    main()
