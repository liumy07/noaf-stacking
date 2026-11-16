# R_cohort_construction — 队列构建（R）

本目录只负责把 **MIMIC-IV / eICU-CRD 导出的原始宽表**整理成**插补前**的分析数据集：

| 脚本 | 输入 | 输出 | 例数 |
|---|---|---|---|
| `01a_mimic_preimputation.R` | `mimic.csv` | `插补前mimic_data.csv`（52 变量） | 34,384（NOAF 9,535 / 非 NOAF 24,849） |
| `01b_eicu_preimputation.R` | `eicu.csv` | `插补前eicu_data.csv`（45 变量） | 32,067（NOAF 2,457 / 非 NOAF 29,610） |

做的事：race 归为 5 类、血管活性药/抗生素/CRRT 指示列合并、`heart_rhythm_first` 文本匹配 + `custom_AF` 校正生成 `NOAF`、剔除缺失 >30% 的列、分类变量转 factor。
**不做**的事：缺失值插补——插补在 Python 终版流程中逐折内部完成（`../01_core_pipeline_v90/step53d_v90_final.py`）。

运行：R ≥ 4.1；改脚本顶部 `DATA_DIR` 与 `OUT_DIR` 两行即可。原始 `mimic.csv` / `eicu.csv` 受 PhysioNet 数据使用协议限制，不随代码分发。
