# =============================================================================
# 01b_eicu_preimputation.R
# -----------------------------------------------------------------------------
# 用途：把从 eICU-CRD 导出的原始宽表（eicu.csv）整理成**插补前**的分析数据集
#       插补前eicu_data.csv（32,067 例 × 45 变量），作为外部验证队列输入。
#
# 本脚本做的事（队列构建 / 变量派生，不含建模）：
#   1. 4 个抗生素指示列 → Antibiotic；4 个血管活性药指示列 → Vasopressors
#   2. 排除既往房颤/房扑（custom_atrial == 0），custom_AF → NOAF
#   3. 删除缺失 >30% 的列（temperature/sbp/dbp/so2/cpk/bnp 等）
#   4. 分类变量转 factor、删除 uniquepid 后导出
#
# 明确**不做**的事：缺失值插补（原因同 01a，初版 missForest 两行保留在文末注释）。
#
# 运行：R >= 4.1，依赖 dplyr / tidyverse
# =============================================================================

# ---- 路径配置（复现时只需改这两行）------------------------------------------
DATA_DIR <- "D:/BJND-4002-11项目/BJND-4002-11/data"   # 存放 mimic.csv / eicu.csv
OUT_DIR  <- "D:/BJND-4002-11项目/BJND-4002-11"        # 输出 插补前*.csv
# -----------------------------------------------------------------------------

library(tableone)

library(survey)
library(foreign)
library(plyr)
library(dplyr)#rename
library(foreign)    
library(dplyr)
library(purrr)
library("survey") 
library("tidyverse") 
library("broom")
library(haven)
library(data.table)
library(lubridate)
### 输入
clearedData <- read.csv(file.path(DATA_DIR, "eicu.csv"), check.names = F)
print(colMeans(is.na(clearedData))*100)
table(clearedData$ethnicity)

# 36636
clear_data=clearedData
colnames(clear_data)
# 步骤1：定义需要判断的原始列名（便于管理和修改）
vaso_cols <- c("custom_Piperacillin", "custom_Vancomycin", "custom_Meropenem", 
               "custom_Cephalosporin")

# 步骤2：新建Vasopressors列 - 任一列为1则标记Yes，否则No
# 行求和法：计算每行1的个数，>0则为Yes，否则No（简洁高效）
clear_data$Antibiotic <- ifelse(
  rowSums(clear_data[, vaso_cols], na.rm = TRUE) > 0,  # na.rm=TRUE避免缺失值干扰
  "Yes", 
  "No"
)

# 步骤3：删除原始的血管加压药列
clear_data <- clear_data[, !names(clear_data) %in% vaso_cols]

# 验证结果：查看新列的分布
table(clear_data$Antibiotic)



colnames(clear_data)
# 步骤1：定义需要判断的原始列名（便于管理和修改）
vaso_cols <- c("custom_DOPam", "custom_Norepinephrine", "custom_Vasopressin", 
               "custom_Norepinephrine.1")

# 步骤2：新建Vasopressors列 - 任一列为1则标记Yes，否则No
# 行求和法：计算每行1的个数，>0则为Yes，否则No（简洁高效）
clear_data$Vasopressors <- ifelse(
  rowSums(clear_data[, vaso_cols], na.rm = TRUE) > 0,  # na.rm=TRUE避免缺失值干扰
  "Yes", 
  "No"
)

# 步骤3：删除原始的血管加压药列
clear_data <- clear_data[, !names(clear_data) %in% vaso_cols]

# 验证结果：查看新列的分布
table(clear_data$Vasopressors)


########删除以往患病######
colnames(clear_data)
table(clear_data$custom_AF)
table(clear_data$custom_atrial,clear_data$custom_AF)
clear_data=clear_data[clear_data$custom_atrial==0,]
# 32067

clear_data$NOAF <- factor(clear_data$custom_AF,levels = c(1,0),labels = c("Yes","No"))
clear_data$custom_AF <- NULL
clear_data$custom_atrial <- NULL

# 验证结果：查看NOAF列分布及数据量
cat("NOAF列最终分布：\n")
print(table(clear_data$NOAF))
cat("\n数据处理后总行数：", nrow(clear_data), "\n")
# Yes    No 
# 2457 29610 
#  32067
print(colMeans(is.na(clear_data))*100)
# 步骤1：计算每列的缺失值占比（百分比）
na_percent <- colMeans(is.na(clear_data)) * 100

# 步骤2：筛选出缺失值占比>30%的列名
high_na_cols <- names(na_percent)[na_percent > 30]

# 步骤3：输出结果（列名+对应缺失占比）
cat("缺失值占比>30%的列：\n")
print(high_na_cols)

# 可选：输出这些列的具体缺失占比，更直观
cat("\n缺失值占比>30%的列及具体占比：\n")
print(na_percent[high_na_cols])
# temperature_max         sbp_max         dbp_max         so2_max         cpk_max      cpk_mb_max         bnp_max temperature_min         sbp_min 
# 86.57810        69.15209        69.15209        48.46415        75.36408        85.36190        86.04484        86.57810        69.15209 
# dbp_min         so2_min         cpk_min      cpk_mb_min         bnp_min 
# 69.15209        48.46415        75.36408        85.36190        86.04484 
# 步骤3：删除这些高缺失列
clear_data <- clear_data[, !names(clear_data) %in% high_na_cols]




# library(missForest)  # 终版不再在 R 中插补，见文末注释
str(clear_data)
# 步骤1：定义需要转换为因子的列名向量
# 步骤1：重新定义【准确匹配数据】的因子列名向量
factor_cols <- c(
  # 字符型列
  "Gender", "ethnicity", "NOAF", "Antibiotic", "Vasopressors",
  # 整型二分类列（注意大小写/拼写）
  "invasiveVent", "sepsis3", "cerebrovascular_disease", 
  "renal_disease", "severe_liver_disease", "mild_liver_disease",
  "congestive_heart_failure", "diabetes", "hypertension"
)

# 步骤2：批量转换为因子（先检查列名是否存在，避免报错）
# 筛选出数据中实际存在的列（防止漏改/错改）
valid_cols <- factor_cols[factor_cols %in% colnames(clear_data)]
clear_data[valid_cols] <- lapply(clear_data[valid_cols], as.factor)

# 步骤3：验证转换结果（查看指定列的类型）
cat("已转换为因子的列及类型：\n")
sapply(clear_data[valid_cols], class)

clear_data$uniquepid <- NULL

# ---- 导出：交给 Python 终版流程的插补前数据集 -------------------------------
write.csv(clear_data, file = file.path(OUT_DIR, "插补前eicu_data.csv"))
cat("written:", file.path(OUT_DIR, "插补前eicu_data.csv"), " rows:", nrow(clear_data), "\n")

# ---- 以下为初版做法（已废弃，仅溯源，勿执行）--------------------------------
#   clear_data <- missForest(clear_data)$ximp
#   write.csv(clear_data, file = "eicu_data.csv")      # 32,067 行（插补后）
# 终版：插补移至 Python，每折内部重拟合（step53d_v90_final.py）
# -----------------------------------------------------------------------------

