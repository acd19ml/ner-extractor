# CS5489 Project - NER Extractor Report: DistilBERT-CRF

**Date:** \today

## Section 1 – Introduction

### Project Goals & Scope
As part of the CS5489 Machine Learning project, this module focuses on the **DistilBERT-CRF** implementation. The objective is to develop a lightweight yet high-performance Named Entity Recognition (NER) system that balances computational efficiency with structured prediction capabilities.

### Dataset & Challenges
The model is trained and evaluated on the CoNLL-2003 English dataset. Key challenges addressed in this implementation include:
*   **Structured Prediction**: Ensuring valid label sequences (e.g., I-PER cannot follow B-ORG) using a CRF layer.
*   **Subword Alignment**: Handling the mismatch between BERT-based subword tokenization and word-level NER labels.
*   **Training Stability**: Mitigating catastrophic forgetting and overfitting through advanced regularization techniques.

#### Data Characteristics
Understanding the data distribution informed our model configuration. The entity classes show natural imbalance, and the sentence length distribution supports our choice of `max_len=192`.

![Entity Frequency](../analysis/figures/entity_frequency.png)
*Figure: Frequency of entity types (PER, LOC, ORG, MISC).*

![Sentence Length Distribution](../analysis/figures/sentence_length_distribution.png)
*Figure: Distribution of sentence lengths in the dataset. Max length was set to 192 to cover the majority of samples.*

## Section 2 – Methodology

### Model Architecture
The DistilBERT-CRF model integrates a pre-trained Transformer encoder with a probabilistic graphical model:

1.  **Encoder (DistilBERT)**: The `distilbert-base-cased` model serves as the backbone, providing contextualized embeddings. We use the **first subword** of each word to represent the token, masking subsequent subwords to maintain alignment with original word-level labels.
2.  **Emission Layer**: A linear projection maps the hidden states to the tag space (BIOES schema).
3.  **Decoder (Linear-Chain CRF)**: A CRF layer models the transition probabilities between adjacent tags. During training, we maximize the log-likelihood of the gold label sequence. During inference, the Viterbi algorithm is used to decode the most probable sequence, enforcing structural constraints (e.g., hard constraints on invalid transitions like `O` -> `I-PER`).

### Training Strategies
To achieve state-of-the-art performance and stability, the following strategies were implemented:

*   **Differential Learning Rates (Diff-LR)**:
    *   Encoder: $2 \times 10^{-5}$ (preventing destruction of pre-trained features).
    *   CRF/Head: $2 \times 10^{-3}$ (fast convergence for task-specific layers).
*   **Layer-wise Learning Rate Decay (LLRD)**: A decay factor of $\gamma=0.95$ is applied, assigning lower learning rates to bottom layers.
*   **R-Drop (Consistency Regularization)**: We enforce KL-divergence consistency between the outputs of two forward passes with dropout ($ \lambda=0.5 $), promoting smoother decision boundaries.
*   **Exponential Moving Average (EMA)**: Model weights are tracked with a decay of $0.999$, and the EMA weights are used for evaluation to improve generalization and robustness.

## Section 3 – Experiments

### Experimental Setup
*   **Cross-Validation**: A **5-Fold Cross-Validation** scheme is used. Crucially, we employ **GroupKFold** grouped by `doc_id` to ensure strict separation between training and validation contexts.
*   **Metric**: The primary metric is **Entity-level Micro-F1** (CoNLL strict matching).
*   **Environment**: Experiments were conducted with mixed-precision (FP16/BF16) and gradient accumulation.

### Hyperparameters
| Parameter | Value |
| :--- | :--- |
| **Model** | `distilbert-base-cased` |
| **Max Sequence Length** | 192 |
| **Batch Size** | 32 (effective) |
| **Optimizer** | AdamW |
| **Learning Rates** | Encoder: $2e-5$, Head: $2e-3$ |
| **Scheduler** | Cosine with 10% Warmup |
| **Regularization** | CRF L2: $1e-4$, Emission Dropout: 0.2 |
| **R-Drop** | $\lambda = 0.5$ |
| **EMA Decay** | 0.999 |

### Results

#### 5-Fold Cross-Validation Results (Stabilization, Augmentation On)
Latest CV sweep (GroupKFold by doc) with Diff-LR + LLRD + R-Drop + EMA + gradual unfreeze + entity-aware augmentation:

| Fold | Validation F1 |
| :---: | :---: |
| 1 | 92.70% |
| 2 | 92.01% |
| 3 | 92.26% |
| 4 | 92.59% |
| 5 | 94.61% |
| **Mean** | **92.84% $\pm$ 1.02%** |

#### Ablation (5-Fold CV Mean F1)
| Variant | Mean F1 | Std |
| :--- | :---: | :---: |
| Main (stabilization) | 92.84% | 0.92% |
| EMA Off | 92.63% | 0.97% |
| R-Drop Off | 92.83% | 0.84% |
| Augmentation On | **92.84%** | 0.93% |

*(All ablations reuse the same splits; values from `training_logs/kfold_results.csv`, primary metric = entity-level micro-F1.)*

#### Final Test Performance
Using the best CV variant (Augmentation On) and training on train+dev, the final model achieves:

* **Validation F1 (full fit)**: 94.27%
* **Test F1**: **89.64%** (precision 88.77%, recall 90.53%, accuracy 97.96%)

This improves the earlier baseline test F1 (89.59%) while retaining the stabilized training recipe.

##### Baseline vs Final (Full Fit)
| Run | Validation F1 | Test Precision | Test Recall | Test F1 |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (distilbert_crf_full) | 94.38% | 89.19% | 90.00% | 89.59% |
| Final (aug on, full fit) | 94.27% | 88.77% | 90.53% | **89.64%** |

*Note*: Baseline reflects the original full-train run; Final uses the stabilized + augmentation recipe that yielded the best CV mean.

## Section 4 – Analysis & Discussion

### Performance Analysis
The DistilBERT-CRF model achieves a Test F1 of ~89.6%, establishing a solid yet improvable baseline. The CRF layer enforces label consistency (BIOES constraints) and mitigates illegal transitions, but the modest test gain over the earlier baseline (≈+0.05 F1) indicates that stabilization and augmentation have limited headroom on this dataset. The 5-fold CV results show low dispersion (std ≈1%), yet the standout Fold 5 hints at distributional sensitivity; per-doc grouping reduces but does not eliminate split-induced variability.

### Convergence & Stability
The integration of R-Drop and EMA contributes to training stability. As shown in the figures below, the training loss decreases steadily without significant fluctuations, and the validation F1 converges and stabilizes around the 10th epoch, preventing overfitting.

![Training Loss Curve](../analysis/figures/training_loss_curve.png)
*Figure: Training Loss over steps. The loss shows consistent convergence.*

![Validation Metrics Curve](../analysis/figures/validation_metrics_curve.png)
*Figure: Validation F1, Precision, and Recall over evaluation steps.*

### Limitations and Threats to Validity
- **CV–Test Gap**: The gap between CV mean (≈92.8% val F1) and test F1 (≈89.6%) suggests residual overfitting to validation folds or distribution mismatch; further regularization or more conservative early stopping may be warranted.
- **Augmentation Impact**: Entity-aware augmentation yields negligible aggregate gains and degrades test F1 on fold checkpoints, implying the current replacement strategy may inject noise; richer constraints or type-aware sampling should be explored.
- **Resource Constraints**: Runs were CPU/MPS-bound; GPU experiments could alter convergence speed and stability. Hyperparameters (e.g., unfreeze schedule) were tuned under this constraint and may not transfer.
- **Metric Coverage**: Illegal-sequence rates and per-entity breakdowns were not logged in the CV tables; this limits our ability to attribute gains to boundary correctness vs. class discrimination.
- **External Validity**: Results are specific to CoNLL-2003; transferability to other domains (noisy text, longer contexts) is unverified.

### Future Work
- Log and report illegal-sequence rates and per-entity F1 for all folds to better attribute improvements to CRF vs. regularizers.
- Refine augmentation (lexical similarity, context-aware constraints) or down-weight it further to avoid noise, then re-run ablations.
- Run GPU-backed sweeps to test sensitivity of gradual unfreezing and LLRD schedules to faster convergence.
- Add error buckets and confusion analyses to quantify boundary vs. type errors; include PCA/UMAP visualizations to assess representation drift under stabilization tricks.
- Benchmark on an out-of-domain dataset to evaluate robustness and generalizability.
