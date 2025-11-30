# DistilBERT-CRF NER 实施计划

1) 目标与交付物 | Objectives & Deliverables
	•	目标：以 DistilBERT-CRF 为主线，完成方法、CV 调优、可视化、消融、报告与基准提交。
Objective: Deliver DistilBERT-CRF with CV tuning, visualizations, ablations, report, and benchmark submissions.
	•	交付：report.pdf、project_code.ipynb、training_logs/、results_summary.csv、leaderboard.txt、可复现实验脚本。
Deliverables: report.pdf, project_code.ipynb, logs, summaries, leaderboard, and reproducible scripts.

⸻

2) 数据与切分 | Data & Splits
	•	标注规范：统一 BIOES；落盘 labels.json；训练前校验标签合法性与“非法序列率”。
Labeling: adopt BIOES; persist labels.json; pre-train checks for label legality & illegal sequence rate.
	•	切分策略：GroupKFold(n_splits=5, groups=doc_id)；公开汇报使用 fold 平均 + 方差。
Splits: GroupKFold(5, groups=doc_id); report mean±std across folds.
	•	统计探索：类频、句长、实体跨度长短、非法序列率；输出至 analysis/figures.
EDA: class freq, sentence length, span length, illegal rate; save plots.

⸻

3) 词级对齐与管线 | Word-level Alignment & Pipeline
	•	对齐规则：保留每词 首个子词 隐向量作为“词表示”；其余子词在 loss/解码中 mask。
Alignment: take the first subword as word representation; mask others in loss/decoding.
	•	评测一致：训练、解码、评测全在词级；仅为可视化将预测回填到原 token 序列。
Consistency: train/decode/eval at word level; map back only for display.

⸻

4) 模型设计 | Model Design
	•	主干：DistilBertModel → 线性发射层（emission） → CRF（Viterbi 解码）。
Backbone: DistilBertModel → linear emissions → CRF (Viterbi).
	•	BIOES + 硬约束：初始化构建 allowed_transitions，对非法转移将 A[i,j] = -1e4。
BIOES + hard constraints: build allowed_transitions; set illegal transitions to -1e4.
	•	先验融合（可选）：Gazetteer 匹配生成 logits，作为 emission 的可学习偏置 emission += α * gazetteer_logits。
Gazetteer prior (opt.): add learnable bias α * gazetteer_logits to emissions.
	•	字符特征（可选轻量）：Char-CNN 32–64 维，与词表示 concat 后线性投影回隐藏维。
Char features (opt.): tiny Char-CNN 32–64d, concat then project back.
	•	正则：emission dropout=0.1–0.3；CRF 转移 L2=1e-4。
Regularization: emission dropout 0.1–0.3; CRF transition L2=1e-4.

⸻

5) 训练与调优 | Training & Tuning
	•	优化器与调度：AdamW；warmup_ratio=0.1 → cosine；weight_decay=0.01；max_grad_norm=1.0。
Optimizer & schedule: AdamW; warmup 10% → cosine; wd=0.01; grad clip=1.0.
	•	差分 LR + LLRD：encoder lr ∈ [1e-5, 5e-5]，head/CRF lr ∈ [1e-4, 5e-3]；LLRD ~0.95/层。
Diff-LR + LLRD: encoder lr [1e-5,5e-5], head/CRF [1e-4,5e-3]; LLRD≈0.95/layer.
	•	渐进解冻：先训 head + 顶层 2–3 层，随后每 1–2 个 epoch 向下解冻。
Gradual unfreezing: head + top 2–3 layers first, unfreeze downwards every 1–2 epochs.
	•	R-Drop（一致性）：同批两次前向（保留 dropout），对 emission logits 加 KL（λ=0.5–1.0）。
R-Drop: two stochastic forwards; KL on emissions (λ=0.5–1.0).
	•	EMA：动量 0.999，验证与选模使用 EMA 权重。
EMA: momentum 0.999; validate/select with EMA weights.
	•	AMP & 累积：bf16/fp16 混合精度；梯度累积稳定等效 batch。
AMP & accumulation: bf16/fp16; use grad accumulation to reach target batch.
	•	数据增强（实体感知）：同类替换 + 上下文保持，对增强样本设 loss 权重 0.5。
Entity-aware augmentation: type-consistent replace; down-weight augmented loss to 0.5.
	•	CV 调参空间（优先级）
Tuning space (prioritized)
	•	max_len ∈ {128, 192, 256}
	•	batch_size ∈ {16, 32}（不足时用累积） / use accumulation if needed
	•	lr_encoder ∈ {1e-5, 2e-5, 3e-5}
	•	lr_head ∈ {5e-4, 1e-3, 2e-3}
	•	crf_L2 ∈ {0, 1e-5, 1e-4, 5e-4}
	•	dropout_emission ∈ {0.1, 0.2, 0.3}
	•	早停与重训：监控 entity-level micro-F1 (strict)，patience 5–8；选最优配置后在 train+dev 全量重训一次并固定随机种子。
Early stop & final fit: monitor entity micro-F1 (strict), patience 5–8; refit once on train+dev with fixed seed.

⸻

6) 评测与分析 | Evaluation & Analysis
	•	主指标：entity-level micro-F1（strict）；同时报 macro-F1、per-class F1、非法序列率。
Primary: entity micro-F1 (strict); also macro-F1, per-class, illegal rate.
	•	错误分桶：边界错误 / 类型混淆 / 非法序列，每类抽样 10 条，附原句+gold+pred。
Error buckets: boundary vs type vs illegal; 10 samples each with context.
	•	表征可视化：固定采样子集，PCA/UMAP + KMeans（k=实体类别数），图例按实体类着色。
Representation viz: fixed subset; PCA/UMAP + KMeans (k=#classes).
	•	不确定性与集成（可选）：MC-Dropout×10 多数投票；或 5-fold span-level 投票集成。
Uncertainty & ensembling (opt.): MC-Dropout×10 majority; or 5-fold span-level ensemble.

⸻

7) 工程与可复现 | Engineering & Reproducibility
	•	长度分桶 + 动态 padding：加速训练、降低显存波动。
Length bucketing + dynamic padding: faster, more stable memory.
	•	配置与快照：YAML 固定随机种子、标签集、BIOES、非法转移掩码开关、LLRD、R-Drop、EMA；训练时保存 allowed_transitions.json。
Config & snapshots: YAML includes seed, labels, BIOES, constraints, LLRD, R-Drop, EMA; persist allowed_transitions.json.
	•	日志与检查点：best.ckpt 与 ema_best.ckpt 双保存；记录 lr/grad_norm/F1 曲线到 CSV/JSONL。
Logging & ckpts: save both raw best and ema_best; log lr, grad_norm, F1 to CSV/JSONL.
	•	Accelerate & Determinism：accelerate 管理多卡；torch.backends.cudnn.deterministic=True，固定 PYTHONHASHSEED。
Accelerate & determinism: use accelerate; set deterministic flags and seeds.

⸻

8) Benchmark / Kaggle
	•	数据适配：统一到 BIOES 与词级对齐；必要时重映射标签。
Adaptation: normalize to BIOES + word-level alignment; remap labels if needed.
	•	提交流程：预测→恢复到 token/原文本格式→产出官方评测/提交文件；记录分数与截图至 leaderboard.txt 与 analysis/benchmarks/。
Submission: predict→map back to token/text→emit official files; store scores & screenshots.

⸻

9) 排期（两周样例） | Timeline (2-week sample)
	•	D1–2：数据清洗/EDA、BIOES 统一、非法率统计；GroupKFold 切分。
D1–2: EDA, BIOES unification, illegal rate; GroupKFold splits.
	•	D3–4：词级对齐实现；DistilBERT-CRF（BIOES+约束）；基础训练可跑通。
D3–4: word-level alignment; DistilBERT-CRF with constraints; base training.
	•	D5–6：5-Fold 调参（Diff-LR、LLRD、max_len、crf_L2、dropout）；早停与日志。
D5–6: 5-fold tuning; early stop; logging.
	•	D7：R-Drop、EMA、渐进解冻。
D7: R-Drop, EMA, gradual unfreezing.
	•	D8：实体增强 + 词典先验；小规模消融对比。
D8: augmentation + gazetteer; ablations.
	•	D9：表征可视化、错误分析模板固定与导出。
D9: viz & error analysis exports.
	•	D10：Benchmark/Kaggle 提交与记录。
D10: benchmark submissions & logging.
	•	D11–12：最终重训、汇总表、报告撰写与整理。
D11–12: final refit, summaries, report.

⸻

10) 风险与缓解 | Risks & Mitigations
	•	类不平衡：分层抽样、损失加权、实体感知增强。
Class imbalance: stratified sampling, loss weights, entity-aware aug.
	•	算力受限：AMP、累积梯度、动态长度裁剪；必要时 max_len=128/192。
Resource limits: AMP, accumulation, dynamic length; shorten max_len.
	•	边界错误：BIOES+硬约束、词级 CRF、R-Drop。
Boundary errors: BIOES+constraints, word-level CRF, R-Drop.
	•	泄漏风险：严格按 doc_id 分组切分；评测只在最终最优模型上一次性跑测试集。
Leakage: GroupKFold by doc; evaluate test once with final model.

⸻

11) 目录结构与配置示例 | Structure & Config Snippet

project/
├─ data/{raw,processed,splits,aug}/
├─ src/
│  ├─ datasets.py        # 词级对齐、动态分桶、增强
│  ├─ modeling.py        # DistilBERT + emissions + CRF(约束)
│  ├─ trainer.py         # EMA、R-Drop、渐进解冻、LLRD
│  ├─ metrics.py         # entity micro/macro-F1, illegal rate
│  └─ utils.py           # seeds, logging, mask builder
├─ configs/
│  ├─ base.yaml
│  ├─ lrdrop_ema.yaml
│  └─ ablation/*.yaml
├─ scripts/{train.sh, eval.sh, kfold.sh, visualize.sh}
├─ analysis/{figures, reports, benchmarks}
├─ training_logs/
└─ notebooks/project_code.ipynb

configs/base.yaml（示例 / example）

seed: 42
schema: BIOES
labels_path: data/labels.json
use_transition_constraints: true

model:
  name: distilbert-base-cased
  dropout_emission: 0.2
  use_char_cnn: false
  use_gazetteer: false

train:
  max_len: 192
  batch_size: 32
  grad_accum_steps: 2
  amp_dtype: bf16
  max_epochs: 20
  early_stop_patience: 6

optim:
  encoder_lr: 2e-5
  head_lr: 2e-3
  weight_decay: 0.01
  lr_schedule: cosine
  warmup_ratio: 0.1
  max_grad_norm: 1.0
  llrd_gamma: 0.95

regularization:
  crf_l2: 1e-4
  rdrop_lambda: 0.5
  use_ema: true
  ema_decay: 0.999

cv:
  n_splits: 5
  group_key: doc_id


⸻

12) 基线与超参建议 | Baseline & Hyperparams
	•	Baseline：DistilBERT-Linear（无 CRF）、max_len=192、encoder_lr=2e-5、dropout=0.1。
Baseline: DistilBERT-Linear (no CRF), max_len=192, encoder_lr=2e-5, dropout=0.1.
	•	主线（推荐）：DistilBERT-CRF（BIOES+约束），encoder_lr=2e-5、head_lr=2e-3、LLRD=0.95、crf_L2=1e-4、dropout_emission=0.2、EMA=0.999、R-Drop λ=0.5、max_len=192。
Mainline (recommended): DistilBERT-CRF with constraints; params as above.

⸻

13) 消融矩阵 | Ablation Matrix
	1.	无 CRF → +CRF（BIOES+约束）
	2.	+LLRD/差分 LR
	3.	+R-Drop
	4.	+EMA
	5.	+Gazetteer
	6.	+Char-CNN
→ 每步记录 entity micro-F1、非法序列率、训练时长与显存峰值。
→ Track entity micro-F1, illegal rate, wall time, and peak memory at each step.

⸻

14) 报告结构要点 | Report Outline Tips
	•	Method：词级对齐示意图、CRF 公式（对数似然、Viterbi）、约束矩阵描述。
Method: word-level alignment diagram, CRF likelihood/Viterbi, constraints.
	•	Experiments：GroupKFold 流程、超参表、消融闭环曲线。
Experiments: GroupKFold flow, hyperparam table, closed-loop ablations.
	•	Analysis：错误分桶表、UMAP/PCA 聚类图、主要混淆对（LOC↔ORG, PER↔ORG）。
Analysis: error buckets, UMAP/PCA clusters, key confusions.
	•	Discussion：CRF 对边界的贡献、R-Drop/EMA 的稳健性、先验/字符特征的场景价值。
Discussion: CRF for boundaries, robustness from R-Drop/EMA, value of priors/char features.

⸻

15) 近期执行清单 | Immediate Action Items
	1.	离线准备 ✅：`models/hf_cache/distilbert-base-cased/` 已就绪，config 指向本地缓存。
	2.	Sanity + 全量训练 ✅：`sanity_check` 与 `distilbert_crf_full` 跑通，validation F1≈0.944，test F1≈0.896。
	3.	结果归档 ➜ `results_summary.csv` 已追加验证/Test 指标，后续实验沿用此文件。
	4.	可视化脚本 ✅：`analysis/scripts/plot_metrics.py`、`entity_stats.py` 生成 loss/F1 曲线与实体/句长图表。
	5.	Notebook 规划（进行中）：在 `notebooks/` 中建骨架 notebook（数据探索 + 训练复现 + 可视化），完善图表输出。
	6.	特征扩展设计：梳理字符嵌入、gazetteer soft labels、FGM/LoRA 实现细节及实验矩阵，补充到 `configs/` 与 `docs/implementation_notes.md` 中。
	7.	实体增强实现 ✅：`src/augmentation.py` + `data_module` 中加入实体感知替换、loss weight（默认 0.5），通过 YAML `augmentation.*` 控制；日志输出增强样本数量，训练日志/Trainer 支持 per-sample loss_weight。
	8.	5-fold 基础设施 ✅：已跑完主线 + 消融（EMA off / R-Drop off / Aug on）5 折，均值已写入 `results_summary.csv`；`training_logs/kfold_results.csv` 记录所有折；后续选择最佳策略做全量 train+dev + test 评估，并在报告中对比均值/方差。

⸻

16) 里程碑划分 | Milestones
	•	M1 – 基线落地（Baseline Delivery）✅  
	  - 内容：数据处理与切分、DistilBERT-CRF 主干实现、单次全量训练、日志与核心可视化；确保 `results_summary.csv` 有 baseline 行，Notebook 输出基础统计；附带最小报告（指标表 + 图表引用 + 结论）与 `data/labels.json`、复现脚本 (`scripts/train_baseline.sh`, `scripts/eval_baseline.sh`)。  
	  - 当前状态：已完成，构成 NER 可交付基线，后续里程碑失败亦可独立交付。
	•	M2 – 训练策略与稳定性（Training Stabilization）  
	  - 内容：差分 LR/LLRD、渐进解冻、R-Drop、EMA、实体感知增强、5-fold 调参，记录各策略对比结果。**当前进展**：diff-LR/LLRD、R-Drop、EMA、渐进解冻已在 `configs/default.yaml` 中预设并通过 `sanity_m2` 验证；下一步聚焦实体增强与 5-fold 框架。  
	  - 目标：在 M1 基线基础上提升训练稳定性与 F1，同时产出调参记录。
	•	M3 – 表征增强（Representation Enhancements）  
	  - 内容：实现 char 特征、gazetteer 融合、BIOES 约束、CRF L2 等；配套可视化/错误分析展示特征贡献。  
	  - 目标：增强模型对边界与类别区分能力。
	•	M4 – 综合优化与基准提交（Advanced Optimization & Benchmark）  
	  - 内容：对抗训练、LoRA/PEFT、MC-Dropout/ensemble、Kaggle 或公开 benchmark 提交；整理最终最佳方案。  
	  - 目标：在外部评测中验证方案效果，形成最终报告亮点。
