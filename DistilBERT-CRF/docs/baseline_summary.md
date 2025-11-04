# DistilBERT-CRF Baseline Summary

## Overview
- Dataset: CoNLL03 (train=13,832 sentences, validation=3,459, test=3,453).
- Model: DistilBERT encoder + linear emission + CRF decoder (word-level alignment).
- Config: `configs/default.yaml` (max_seq_length=256, batch_size=16, lr=3e-5, epochs=10).
- Training command:
  ```bash
  python scripts/train_distilbert_crf.py \
      --config configs/default.yaml \
      --run-name distilbert_crf_full
  ```
- Hardware: CPU (AMP disabled), runtime ≈ 4.5 h.

## Metrics
| Split | Precision | Recall | F1 | Accuracy | Loss | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| Validation | 0.9414 | 0.9463 | 0.9438 | – | – | best |
| Test | 0.8919 | 0.9000 | 0.8959 | 0.9794 | 1.8198 | best |

Detailed entries stored in `results_summary.csv`.

## Visualizations
- `analysis/figures/training_loss_curve.png`: training loss vs. step.
- `analysis/figures/validation_metrics_curve.png`: validation precision/recall/F1 progression.
- `analysis/figures/entity_frequency.png`: overall entity distribution.
- `analysis/figures/sentence_length_distribution.png`: length statistics across splits.

## Error Snapshot
Notebook cell “Error Analysis” lists top-5 misclassified entities with sentence context, highlighting major type-confusion cases (e.g., ORG↔LOC, LOC↔O).

## Artifacts
- Checkpoint: `models/distilbert_crf/distilbert_crf_full/best/` (config + safetensors + label map).
- Training log: `training_logs/distilbert_crf_full.log`.
- Repro scripts: `scripts/train_baseline.sh`, `scripts/eval_baseline.sh`.
- Labels: `data/labels.json` (BIO labels in training vocabulary order).

