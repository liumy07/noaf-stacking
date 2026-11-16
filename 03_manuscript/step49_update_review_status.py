# -*- coding: utf-8 -*-
"""
step49: update the 修改状态 (col H) and 学生备注/完成证据 (col I) columns of the
two second-round reviewer workbooks, after the v80 manuscript revision (step48).

The workbooks contain slightly invalid XML that openpyxl refuses to load, so we
edit xl/worksheets/sheet1.xml in place via zipfile + ElementTree and rewrite the
package, copying every other entry byte-for-byte.

Rows: Major  MA01-MA05 -> rows 5-9 ;  Moderate MO01-MO17 -> rows 5-21.
Status column H; evidence column I (inline strings, style attribute preserved).
"""
import zipfile, shutil, xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(r"D:\BJND-4002-11项目\BJND-4002-11")
OUT  = Path(r"C:\Users\localuser\WorkBuddy\2026-09-09-17-02-31\output\rev_noaf")
NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS = "{%s}" % NS_MAIN

MAJOR = {
 "MA01": ("已完成（v80 重跑）",
  "已按建议重跑并改写 Methods 2.3–2.4（稿件 P21/P24–25）：(1) 30% 缺失率删除规则与原始缺失率见 Methods 及补充材料 S8；"
  "(2) missForest 型 IterativeImputer(RandomForestRegressor) 仅在 70% 训练划分上拟合（代码 step20_v70_impute.py，"
  "日志 'fitted on TRAIN ONLY'），内部测试集与 eICU 仅用冻结插补器 transform，从未重新拟合，全队列插补已废弃；"
  "(3) 插补嵌入数据流水线并在 CV 折内不接触任何折外/测试数据。影响已通过 v80 全流程重跑体现（OOF AUC 0.8418，"
  "内测 0.8430，eICU 0.7531）。"),
 "MA02": ("已完成（最高优先级，已重跑）",
  "v80（step40_v80_foldwise.py）已把单因素 logistic（P<0.05）+ LASSO（10 折 CV、1-SE 规则、加强惩罚 fallback）"
  "纳入每个 OOF 训练折内部重新执行：10 折×每折重选特征（v80_fold_audit.csv 逐折记录入选变量数与名单），"
  "OOF 概率严格无泄漏，meta-learner 在 OOF 上训练（v80_scores.npz/v80_selection_audit.csv）。"
  "全训练集锁定最终 26 个变量后重拟合基学习器（OOF AUC：MLP 0.8371 / RF 0.8359 / XGBoost 0.8340，"
  "未四舍五入排序），LASSO meta 权重 2.362/2.169/0.961；内测 0.8430、eICU 0.7531 为冻结模型一次前向预测，"
  "测试集仅在最终模型锁定后评估。Methods 2.4–2.6 与 Results 已相应改写。"),
 "MA03": ("已完成（统一冻结预测器）",
  "v80 采用单一冻结预测函数：全训练集重拟合的 top-3 基学习器 + LASSO meta（step40），内测、eICU、SHAP "
  "全部调用同一 pipeline。step44 断言 rebuilt internal-test AUC 与 0.8430 bit-identical（v80_shap_audit.txt），"
  "解释函数=部署函数。已弃用 10 折模型均值预测；10 折 ensemble 仅作为敏感性对照在 Methods 中说明。"
  "Figure 4–5 均基于同一冻结模型重绘。"),
 "MA04": ("已完成（措辞修正，未重算）",
  "Methods 2.7 与 Results 3.4 的 DCA 阈值解释已改写：阈值概率 pt=0.20 对应损失比 w = pt/(1−pt) = 0.25，"
  "即假定 1 例漏诊的代价相当于 4 例假阳性的 0.25 倍权重；明确 net benefit 以 treat-all 为参照、"
  "DCA 使用冻结模型在两队列的概率与真实 NOAF 标签，'clinically relevant' 措辞已删除并改为"
  "'net benefit at threshold probabilities up to 0.4'。概率、标签与曲线生成代码（step8/step30 系列）复核一致。"),
 "MA05": ("已完成（重复段已删除）",
  "Discussion 4.4 中与 4.3 重复的 4 段旧稿（原 P84–P87：心率、心衰、升压药及瀑布图重复段落）已删除，"
  "保留整合后的精简分析（P82）与非因果 SHAP 解释；全文已通读核查无其他拼接残留（P75–P94 审计通过）。"),
}

MODERATE = {
 "MO01": ("已完成（说明已补充）",
  "Methods 2.3 已补充：eICU-CRD 房颤判定使用 chartevents/d_labitems 中的 custom_AF / custom_atrial 字段"
  "（01_clean.R），以入 ICU 后前 24 小时内任一 AF 记录为阳性，排除既往 AF 与心脏手术后患者；"
  "24 小时时间窗口沿用已发表 ICU NOAF 文献并已引用。适用条件已在人群定义中写明。"),
 "MO02": ("已完成（人群限定；landmark 列为局限）",
  "Results 3.1 与 Methods 2.2 已明确适用人群为 ICU 住院 >48 小时且存活至 48 小时的患者；"
  "Discussion 局限性中已说明由此产生的存活/选择偏倚，24 小时 landmark 敏感性分析因数据字段限制列为未来工作。"),
 "MO03": ("已完成",
  "稿件 P21 与补充材料 S8 已改为报告填补前各数据库/各变量原始缺失率及 >30% 删除规则（v70_impute 日志输出 "
  "na_pct_before）；'missForest' 已更正为 'iterative single imputation（missForest-type IterativeImputer）'，"
  "未声称多重插补；'填补后 39 特征 0% 缺失' 表述已删除。"),
 "MO04": ("已完成（局限性处理）",
  "未新增无单因素筛选的敏感性分析；Methods 2.4 已说明两阶段筛选为预设定策略，Discussion 局限性保留"
  "'可能丢失仅具联合预测信息的变量'，并已删除 '唯一最优特征集' 类措辞（P29/P75）。"),
 "MO05": ("已完成",
  "Methods 2.6 与 Results 3.4 已报告未四舍五入 OOF AUC：MLP 0.8371 / RF 0.8359 / XGBoost 0.8340（v80_results.csv），"
  "排名依据 4 位小数；并列处理规则（OOF AUC 相同则依次比较 AP、Brier）已在 Methods 写明，"
  "top-3 在内测集揭盲前由 OOF 确定（v80_fold_audit.csv 时间戳顺序可证）。"),
 "MO06": ("已完成",
  "Results 3.4 与 Abstract 已同时给出配对 DeLong 绝对差值、95% CI 与 P 值（内测 ΔAUC vs RF 0.0048 "
  "[0.0029–0.0068]、vs XGBoost 0.0052 [0.0027–0.0076]、vs MLP 0.0084 [0.0063–0.0105]，均 P<0.001），"
  "并将 'significant advantage' 改为 'small but consistent improvement in discrimination'（v80_delong_paired.csv）。"),
 "MO07": ("已完成（如实描述）",
  "Methods 2.7 已写明校准截距与斜率由同一 logistic 重校准模型 logit(p) = α + β·logit(p̂) 联合估计"
  "（statsmodels Logit，step2/step41 系列），并非固定斜率 β=1 时单独估计的截距；补充材料 S5 公式与"
  "图表脚注已同步更正（v80_ci_calibration.csv）。"),
 "MO08": ("已完成",
  "内部 calibration slope 1.14（0.918 OOF / 1.137 内测，v80_results.csv）不再称 '近乎完美'；"
  "Results 3.4 与 Abstract 改为 'internal calibration was substantially better than external, "
  "though the slope still deviated modestly from 1'。"),
 "MO09": ("已完成",
  "Results 3.4/Discussion 4.3 已改为总体校准偏差措辞：E:O = 3.02 仅描述队列层面预测事件数与观察事件数之比，"
  "删除了 '0.30 对应 0.10' 的无分层示例；分层数据不支持时不再外推至个体风险倍数。"),
 "MO10": ("已完成",
  "Results 3.1 已改写为 'eICU-CRD 2.6% vs MIMIC-IV 13.5%'（轻度肝病），全文比较顺序已统一为"
  "eICU 在前（P75 前后各处复核）。"),
 "MO11": ("已完成（核查+讨论补充）",
  "已复核两库抗生素相关 SQL/字段映射（01_clean.R：abx 类目、单位换算），Discussion 4.2 已补充：跨库差异"
  "可能部分源于记录实践与映射口径不同，而非单纯 case mix（P80）。"),
 "MO12": ("已完成",
  "Methods 2.7 与 Figure 6 图注已说明：分类变量经 one-hot 编码，'Vasopressor use' 为使用/未使用的统一编码特征，"
  "未将 '未使用' 单独列特征；beeswarm 颜色=特征值高低（低→蓝、高→红），SHAP 值为概率尺度。"),
 "MO13": ("已完成",
  "Methods 2.7 与 Results 3.5 已改称四种病例为 '极端高置信度示例（extreme high-confidence examples）'，"
  "明确判定阈值 0.5 与选择规则（内测集各类型中预测概率最高/最低的 1 例，v80_shap_cases.csv："
  "TP p=0.9127 / TN p=0.0619 / FP p=0.9102 / FN p=0.0632），不再称典型患者。"),
 "MO14": ("待作者补充",
  "Declarations 中 Funding / 作者贡献 / 代码可用性等 [AUTHOR TO COMPLETE] 占位符需作者本人填写真实信息；"
  "投稿前需删除全部占位符并随稿提供补充代码（step 系列脚本已整理至代码归档包）。此条无法由分析侧代填。"),
 "MO15": ("已完成（版本已实测核验）",
  "P35 已按实际执行环境核验更新：Python 3.13.12 / scikit-learn 1.9.1 / XGBoost 3.4.1 / LightGBM 4.7.0 / "
  "SHAP 0.52.0 / statsmodels 0.15.0 / SciPy 1.18.1 / NumPy 2.5.2 / pandas 3.0.6 / matplotlib 3.11.1 "
  "（pip freeze 实测，非估填）。"),
 "MO16": ("已完成（图版已拆分重排）",
  "Figure 4 已由 9-panel 拆分：工作流程图并入 Figure 1 体系；Figure 4（A–H，2×4）为堆叠模型内部验证"
  "ROC/PR/DCA/校准，新增 Figure 5（外部 eICU 同布局）与 Figure 6（SHAP：beeswarm、重要性、稳定性、瀑布图）；"
  "坐标/图例/字号已放大，PIL 组合脚本 step47，单图源文件均为 SVG/PDF 矢量。"),
 "MO17": ("补充材料待随稿提交",
  "补充表 S1–S9（tables/）、补充图（FigS1 训练-OOF ROC/PR、FigS4 DeLong 森林图、Fig6S/Fig7S 内部与外部"
  "单模型曲线等）与 TRIPOD+AI 清单已生成并逐一核对编号与主文数值一致；需在投稿时打包提交，本轮仅交付主文档。"),
}


def set_cell(row_el, col, text):
    """Replace or insert an inline-string cell in the given row element."""
    cell = None
    for c in row_el.findall(NS + "c"):
        ref = c.get("r", "")
        cur = "".join(ch for ch in ref if ch.isalpha())
        if cur == col:
            cell = c
            break
    if cell is None:
        # insert maintaining column order
        cell = ET.Element(NS + "c")
        cell.set("r", f"{col}{row_el.get('r')}")
        row_el.append(cell)
    cell.set("t", "inlineStr")
    for child in list(cell):
        cell.remove(child)
    is_el = ET.SubElement(cell, NS + "is")
    t_el = ET.SubElement(is_el, NS + "t")
    t_el.text = text


def cell_text(c, shared):
    if c is None:
        return ""
    if c.get("t") == "s":
        v = c.find(NS + "v")
        if v is not None and v.text is not None:
            try:
                return shared[int(v.text)]
            except (ValueError, IndexError):
                return ""
        return ""
    is_el = c.find(NS + "is")
    if is_el is not None:
        return "".join(t.text or "" for t in is_el.iter(NS + "t"))
    v = c.find(NS + "v")
    return v.text if v is not None else ""


def update_workbook(path, mapping):
    backup = OUT / ("_backup_" + path.name)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  [BACKUP] {backup.name}")
    z = zipfile.ZipFile(path)
    entries = {n: z.read(n) for n in z.namelist()}
    z.close()
    shared = []
    if "xl/sharedStrings.xml" in entries:
        ss = ET.fromstring(entries["xl/sharedStrings.xml"])
        for si in ss.findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    ET.register_namespace("", NS_MAIN)
    root = ET.fromstring(entries["xl/worksheets/sheet1.xml"])
    sheet_data = root.find(NS + "sheetData")
    changed = 0
    for row in sheet_data.findall(NS + "row"):
        cells = {"".join(ch for ch in c.get("r", "") if ch.isalpha()): c
                 for c in row.findall(NS + "c")}
        key = cell_text(cells.get("A"), shared).strip()
        if key in mapping:
            status, note = mapping[key]
            set_cell(row, "H", status)
            set_cell(row, "I", note)
            changed += 1
            print(f"  [OK] {key}: {status}")
    # shared-string rows would not match by key; verify nothing was missed
    assert changed == len(mapping), f"only {changed}/{len(mapping)} rows updated"
    entries["xl/worksheets/sheet1.xml"] = ET.tostring(
        root, xml_declaration=True, encoding="UTF-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zo:
        for name, data in entries.items():
            zo.writestr(name, data)
    print(f"  [SAVED] {path.name}")


if __name__ == "__main__":
    print("== Major Issues ==")
    update_workbook(BASE / "NOAF_Second_Round_Major_Issues.xlsx", MAJOR)
    print("== Moderate Issues ==")
    update_workbook(BASE / "NOAF_Second_Round_Moderate_Issues.xlsx", MODERATE)
