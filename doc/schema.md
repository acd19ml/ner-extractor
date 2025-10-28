
---

## CS5489 Project – NER Extractor

本文件基于课程评分细则（总分 85）对 NER 项目进行对齐规划，给出可执行方案与交付清单，目标是覆盖各打分点并达到满分区间。

---

## 一、方法探索（Project Results: Methods, 15 pts）

计划至少实现并对比 4 条技术线：
- BiLSTM-CRF（经典序列标注基线）
- DistilBERT-CRF（轻量 Transformer + 条件随机场）
- RoBERTa-CRF（强基线，高性能）
- 多任务/指令微调扩展（可选）：命名实体 + 辅助 span 分类

对齐交付：
- 方法简述、模型结构图、训练超参表
- 每方法训练/验证曲线与分数（Precision/Recall/F1、micro/macro、entity-level F1）

---

## 二、实验设置（Experiment Setup, 10 pts）

- 数据划分：训练/验证（80%/20%）+ 固定随机种子 + 分层抽样（按实体分布）
- 5-Fold Cross-Validation：用于超参调优（学习率、权重衰减、CRF 转移正则、最大序列长度、batch size）
- 全量记录：训练 loss、验证 micro-F1、learning rate schedule、早停轮数
- 最优配置在测试集上锁定评估，仅报告一次测试结果

---

## 三、特征与数据（Project Results: Features 10 pts + Extra: Data 5 pts）

特征设计与对比：
- Token 表示：WordPiece/BPE 对比、cased vs uncased
- 字符级特征：Char-CNN/Char-BiLSTM 与无字符特征对比（在 BiLSTM-CRF 中最显著）
- 词形/词类特征：大小写模式、数字掩码、POS（可选）
- 外部资源：Gazetteer/字典匹配特征（BIO soft labels）
- 上下文表示：静态词向量（GloVe）vs 预训练上下文表征（DistilBERT/RoBERTa）

数据探索与可视化：
- 实体类别频次、句长分布、标签不均衡分析
- 错误分析：BIO 合法性、跨句实体、嵌套/重叠实体占比（若存在）

数据增强（审慎）：
- 同类型实体表内替换（gazetteer-aware replacement）
- 非实体区域的同义改写/插入（避免破坏标注）

交付：统计图（直方图/箱线图）、样例表、特征消融实验对比图。

---

## 四、降维/聚类（DimRed/Clustering, 5 pts）

在验证集抽样句子上，收集 token/实体 span 的隐藏表示：
- 降维：PCA 与 UMAP 各一次
- 聚类：KMeans 按实体类型数 k 聚类
- 可视化：二维散点图，颜色=实体类型/簇，标注代表性样本

交付：嵌入可分性对比图（BiLSTM-CRF vs RoBERTa-CRF），并讨论类间边界与混淆。

---

## 五、Kaggle 或公开 Benchmark（5 pts）

两条路线，择一：
- Kaggle/Leaderboard：将最优模型提交到相关 NER 榜单，目标 Top X%（≥ 2.5 分档），记录排名与分数
- 公开数据集基准：如 CoNLL-2003/OntoNotes 子集，复现实验并与公开结果对比

交付：方法 vs 排名/分数 对比表、提交截图/链接。

---

## 六、项目加分项（Project Extra, 25 pts）

Data（5 pts）：
- 全面可视化与统计；展示增强带来的分布/性能变化

Method（10 pts）：
- 新方法>1：如 CRF 结构化蒸馏、对抗训练（FGM/PGD）、参数高效微调（LoRA）

Features（5 pts）：
- 新表征：字典引导的 soft BIO 先验、提示式 span 表示（prompt tokens）

Justification（5 pts）：
- 为每个新方法提供动机、预期收益、边界条件与失败案例

---

## 七、报告结构（Descriptions 9 + Discussion 3 + Figures 3）

### Section 1 – Introduction
- 任务目标、数据集、挑战点（罕见实体、跨域、歧义）

### Section 2 – Methodology
- 各方法原理图与差异；损失函数（CRF 对数似然）、解码（Viterbi）、训练细节

### Section 3 – Experiments
- 数据划分、交叉验证流程图、超参表、评测指标定义
- 结果表：各方法/特征消融的 P/R/F1（micro/macro/entity-level）
- 曲线：训练/验证损失、F1；收敛/早停对比

### Section 4 – Analysis & Discussion
- 错误类型分析（边界、实体类型混淆、标签非法）
- DimRed/Clustering 可视化解读；为何某法更稳健（如 RoBERTa-CRF）

### Section 5 – Figures
- 统一风格（matplotlib + seaborn），所有图表可复现实验脚本

---

## 八、交付物清单（Final Deliverables）

| 文件名 | 内容 | 格式 |
| --- | --- | --- |
| report.pdf | 完整报告（含图表与深入讨论） | PDF |
| project_code.ipynb | 主实验与可视化（可复现） | Jupyter Notebook |
| training_logs/ | 训练/验证日志（jsonl/csv） | 文本 |
| loss_curves/ | 损失与 F1 曲线图 | PNG |
| results_summary.csv | 方法/折次/分数汇总 | CSV |
| leaderboard.txt | Kaggle/Benchmark 提交记录 | TXT |
| README.md | 环境、训练、评测说明 | Markdown |

---

## 九、评分映射与预期（目标：85/85）

| Rubric 项目 | 预期得分 | 实现要点 |
| --- | --- | --- |
| Methods | 15 | ≥3 种方法，对比与消融 |
| Experiment Setup | 10 | 固定划分，5-fold，最终单次测试 |
| Features | 10 | 字符级/词形/字典/上下文多表征对比 |
| DimRed/Clustering | 5 | PCA/UMAP + KMeans + 可视化 |
| Kaggle | 5 | 提交并达成 Top 档或公开对比 |
| Extra-Data | 5 | 全面可视化与增强分析 |
| Extra-Method | 10 | ≥2 个新方法（如蒸馏/对抗/PEFT） |
| Extra-Features | 5 | 新表征（soft BIO/prompt tokens） |
| Justification | 5 | 动机、边界、失败案例 |
| Report | 15 | 深入描述、讨论与高质量图表 |
| Total | 85/85 | 满分对齐 |

---