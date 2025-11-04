# Notebook Guide

## `project_code.ipynb`
- Mirrors Milestone 1 deliverables: data stats, training log review, evaluation metrics, error cases.
- Uses markdown sections to present figures from `analysis/figures/` (sentence lengths, entity frequency, loss curves).
- Tables embed baseline metrics (validation/test F1, runtime, commands) and sample mistakes.
- Run top-to-bottom after the baseline training (`./scripts/train_baseline.sh`), or rerun evaluation via `./scripts/eval_baseline.sh`.
- Designed as a report-ready artifact; export to HTML/PDF directly if needed.

## Tips
- The notebook assumes processed data under `data/processed/conll03/` and checkpoints in `models/distilbert_crf/`.
- For further experiments (Milestones 2–4), duplicate this notebook and extend with new figures/analysis rather than overwriting the baseline version.
