# =============================================================================
# 01a_mimic_preimputation.R
# -----------------------------------------------------------------------------
# 用途：把从 MIMIC-IV 导出的原始宽表（mimic.csv）整理成**插补前**的分析数据集
#       插补前mimic_data.csv（34,384 例 × 52 变量），作为 Python 终版流程的输入。
#
# 本脚本做的事（全部为队列构建 / 变量派生，不含任何建模）：
#   1. race 原始 30+ 取值合并为 White / Black / Asian / Hispanic / Other
#   2. 6 个血管活性药指示列 → 单一 Vasopressors（任一为 1 即 Yes）
#   3. antibiotic_first 文本 → Antibiotic（非空即 Yes）
#   4. heart_rhythm_first 文本中匹配 "AF (Atrial Fibrillation)" → AF，
#      再用 custom_AF 校正（custom_AF=1 但 AF=No 的行删除）→ NOAF
#   5. 删除缺失 >30% 的列
#   6. 分类变量转 factor 后导出
#
# 明确**不做**的事：缺失值插补。初版在此处调用 missForest() 对全队列插补，
#   该步骤发生在训练/测试拆分之前，会引入预处理泄漏；终版已改为在 Python
#   中每折内部重拟合（见 step53d_v90_final.py）。原插补两行保留在文末注释中，
#   仅作溯源，不要执行。
#
# 运行：R >= 4.1，依赖 dplyr / tidyverse（read.csv 为 base）
# 数据：mimic.csv 由 MIMIC-IV 提取脚本导出，受数据使用协议限制不可再分发
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
clearedData <- read.csv(file.path(DATA_DIR, "mimic.csv"), check.names = F)
print(colMeans(is.na(clearedData))*100)
table(clearedData$race)
clearedData$race <- factor(clearedData$race,levels = c("AMERICAN INDIAN/ALASKA NATIVE","ASIAN","ASIAN - ASIAN INDIAN","ASIAN - CHINESE","ASIAN - KOREAN","ASIAN - SOUTH EAST ASIAN",
                                                       "BLACK/AFRICAN","BLACK/AFRICAN AMERICAN","BLACK/CAPE VERDEAN","BLACK/CARIBBEAN ISLAND","HISPANIC OR LATINO","HISPANIC/LATINO - CENTRAL AMERICAN",
                                                       "HISPANIC/LATINO - COLUMBIAN","HISPANIC/LATINO - CUBAN","HISPANIC/LATINO - DOMINICAN","HISPANIC/LATINO - GUATEMALAN","HISPANIC/LATINO - HONDURAN","HISPANIC/LATINO - MEXICAN",
                                                       "HISPANIC/LATINO - PUERTO RICAN","HISPANIC/LATINO - SALVADORAN",
                                                       "MULTIPLE RACE/ETHNICITY","NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER","OTHER",
                                                       "PATIENT DECLINED TO ANSWER","PORTUGUESE","SOUTH AMERICAN","UNABLE TO OBTAIN","UNKNOWN","WHITE","WHITE - BRAZILIAN","WHITE - EASTERN EUROPEAN",'WHITE - OTHER EUROPEAN',
                                                       "WHITE - RUSSIAN"),
                           labels = c("Other","Asian","Asian","Asian","Asian","Asian","Black","Black","Black","Black","Hispanic","Hispanic","Hispanic","Hispanic","Hispanic","Hispanic","Hispanic","Hispanic","Hispanic","Hispanic",
                                      "Other","Other","Other","Other","Other","Other","Other","Other","White","White","White","White","White"))

table(clearedData$race)
# 37830
clear_data=clearedData
colnames(clear_data)
# 步骤1：定义需要判断的原始列名（便于管理和修改）
vaso_cols <- c("custom_DOBU", "custom_DOPamine", "custom_Vaso", 
               "custom_vasopre", "custom_NORep", "custom_Epinep")

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

# 步骤1：修复无效UTF-8编码字符（核心解决报错问题）
# 方法：将字符串转为ASCII编码，无法转换的字符替换为空格
clear_data$antibiotic_first <- iconv(
  clear_data$antibiotic_first, 
  from = "UTF-8", 
  to = "ASCII", 
  sub = " "  # 无效字符替换为空格
)

# 步骤2：去除字符串两端的空格（替代原trimws，避免报错）
clear_data$antibiotic_first <- gsub("^\\s+|\\s+$", "", clear_data$antibiotic_first)

# 步骤3：新建Antibiotic列 - 非空=Yes，空/NA=No
clear_data$Antibiotic <- ifelse(
  # 同时处理空字符串、NA、仅空格的情况
  nchar(clear_data$antibiotic_first) > 0 & !is.na(clear_data$antibiotic_first),
  "Yes",
  "No"
)

# 步骤4：删除原始的antibiotic_first列
clear_data$antibiotic_first <- NULL

# 验证结果：查看新列分布
table(clear_data$Antibiotic)


########删除以往患病######
# 步骤1：修复潜在的UTF-8编码问题（避免字符串处理报错）
clear_data$heart_rhythm_first <- iconv(
  clear_data$heart_rhythm_first, 
  from = "UTF-8", 
  to = "ASCII", 
  sub = " "  # 无效字符替换为空格
)

# 步骤2：深度清理空白字符（删除首尾空格、换行符、多余空格）
# 1. 去除换行符\n和制表符\t
clear_data$heart_rhythm_first <- gsub("\\n|\\t", "", clear_data$heart_rhythm_first)
# 2. 去除首尾空格
clear_data$heart_rhythm_first <- gsub("^\\s+|\\s+$", "", clear_data$heart_rhythm_first)
# 3. 去除中间多余空格（如“AF (Atrial Fibrillation) ;”中的多余空格）
clear_data$heart_rhythm_first <- gsub("\\s+", " ", clear_data$heart_rhythm_first)

# 步骤3：新建AF列 - 包含AF (Atrial Fibrillation)标Yes，否则No
# grepl模糊匹配，ignore.case=TRUE忽略大小写，确保匹配精准
clear_data$AF <- ifelse(
  grepl("AF \\(Atrial Fibrillation\\)", clear_data$heart_rhythm_first, ignore.case = TRUE),
  "Yes",
  "No"
)

# 步骤4：删除原始的heart_rhythm_first列
clear_data$heart_rhythm_first <- NULL

# 验证结果：查看AF列的分布
table(clear_data$AF)
table(clear_data$custom_AF,clear_data$AF)
# 步骤1：删除"custom_AF=1 且 AF=No"的行（核心删除逻辑）
clear_data <- clear_data[!(clear_data$custom_AF == 1 & clear_data$AF == "No"), ]

# 步骤2：新建NOAF列 - 按规则标记Yes/No
clear_data$NOAF <- ifelse(
  clear_data$AF == "Yes",  # AF为Yes → NOAF=Yes
  "Yes",
  "No"                    # 其余情况（AF=No且custom_AF=0）→ NOAF=No
)

# 步骤3：删除原始列（AF和custom_AF）
clear_data$AF <- NULL
clear_data$custom_AF <- NULL
clear_data$icu_intime <- NULL
# 验证结果：查看NOAF列分布及数据量
cat("NOAF列最终分布：\n")
print(table(clear_data$NOAF))
cat("\n数据处理后总行数：", nrow(clear_data), "\n")
# No   Yes 
# 24849  9535
# 34384 
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
# "so2_max"      "ck_mb_max"    "ntprobnp_max" "ck_cpk_max"   "so2_min"      "ck_mb_min"    "ntprobnp_min" "ck_cpk_min"


# 步骤3：删除这些高缺失列
clear_data <- clear_data[, !names(clear_data) %in% high_na_cols]




# library(missForest)  # 终版不再在 R 中插补，见文末注释
str(clear_data)
# 步骤1：定义需要转换为因子的列名向量
factor_cols <- c(
  "Gender", "Marital_status", "NOAF", "Antibiotic","Vasopressors", "crrt", 
  "InvasiveVent", "sepsis", "cerebrovascular_disease", 
  "renal_disease", "severe_liver_disease", "mild_liver_disease",'congestive_heart_failure',"diabetes","hypertension"
)

# 步骤2：批量转换为因子（一行代码完成所有列）
clear_data[factor_cols] <- lapply(clear_data[factor_cols], as.factor)

# 验证转换结果（可选）：查看指定列的类型
sapply(clear_data[factor_cols], class)

# ---- 导出：交给 Python 终版流程的插补前数据集 -------------------------------
write.csv(clear_data, file = file.path(OUT_DIR, "插补前mimic_data.csv"))
cat("written:", file.path(OUT_DIR, "插补前mimic_data.csv"), " rows:", nrow(clear_data), "\n")

# ---- 以下为初版做法（已废弃，仅溯源，勿执行）--------------------------------
# 初版在拆分前对全队列插补，构成预处理泄漏：
#   clear_data <- missForest(clear_data)$ximp
#   write.csv(clear_data, file = "mimic_data.csv")     # 34,384 行（插补后）
# 终版：插补移至 Python，每一折只用该折训练行拟合 missForest（step53d_v90_final.py）
# -----------------------------------------------------------------------------

