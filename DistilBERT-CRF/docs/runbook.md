# DistilBERT-CRF Runbook

This runbook captures the day-to-day workflow for preparing data, training, and evaluating the DistilBERT-CRF NER model.

## 1. Environment Setup
- Ensure Python 3.13.7 is active: `pyenv local 3.13.7`
- Create and activate the project virtualenv:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Offline environments: pre-download `distilbert-base-cased` (at least `config.json`, `pytorch_model.bin` or `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`) and point `configs/*.yaml` `model.pretrained_model_name` to the local directory, or set `HF_HOME` to a cached mirror before training.

## 2. Data Preparation (CoNLL03)
- Place the raw CoNLL03 archive under `data/raw/conll03/` (e.g., `conll2003.zip`).
- Unpack the archive:
  ```bash
  unzip -q data/raw/conll03/conll2003.zip -d data/raw/conll03
  ```
- Generate normalized train/validation/test splits:
  ```bash
  python scripts/prepare_conll03.py
  ```
- Processed files are written to `data/processed/conll03/{train,validation,test}.txt`.

## 3. Training the Model
- Launch fine-tuning with the default configuration:
  ```bash
  python scripts/train_distilbert_crf.py \
      --config configs/default.yaml \
      --run-name distilbert_crf_baseline
  ```
  或使用包装脚本：
  ```bash
  ./scripts/train_baseline.sh
  ```
- Quick sanity/overfit check (small subset & lightweight hyperparameters):
  ```bash
  python scripts/train_distilbert_crf.py \
      --config configs/sanity.yaml \
      --run-name sanity_check \
      --max-train-samples 128 \
      --max-eval-samples 128
  ```
- Optional flags:
  - `--freeze-encoder-layers N` freezes the lowest `N` DistilBERT layers for stability experiments.
  - `--evaluate-test` runs a final evaluation on the test split after training.
  - `--skip-training --evaluate-test` loads an existing checkpoint (default `best/`) and reports test metrics without re-running fine-tuning.
- Cross-validation: run `scripts/generate_kfold_splits.py` once to produce a doc-aware JSON split (saved under `data/processed/conll03/kfold_splits.json` by default), then call `scripts/run_kfold.py` to orchestrate the five folds. The launcher exports `NER_FOLD_*` env vars so `train_distilbert_crf.py` reuses a single config while swapping subsets and captures per-fold summaries in `training_logs/kfold_results.csv` (configurable via `--results-file`).
- Ablations: use the prebuilt configs under `configs/ablation/` (`ema_off.yaml`, `rdrop_off.yaml`, `aug_on.yaml`) with `scripts/run_kfold.py --config <cfg> --run-prefix <name>` to produce variant CV metrics. Summaries can be computed via `scripts/summarize_kfold.py --run-prefix <name> --csv training_logs/kfold_results.csv`.
- Entity-aware augmentation:
  - Toggle via `augmentation.enabled` in the YAML config. Additional knobs (`entity_replace_prob`, `max_entity_replacements`, `copies_per_sample`, `loss_weight`, `max_generated_samples`, `seed`) control how aggressively spans are replaced and how augmented sentences are weighted in the loss.
  - Augmented samples are appended only to the training split and down-weighted during optimization (default `loss_weight=0.5`) to avoid overpowering real data.
- Checkpoints and logs:
  - Best model: `models/distilbert_crf_baseline/best/`
  - Periodic checkpoints: `models/distilbert_crf_baseline/step_*`
  - Training log: `training_logs/distilbert_crf_baseline.log`

## 4. Evaluating Existing Checkpoints
- Reuse the training script with `--evaluate-test` to evaluate the best saved checkpoint.
- Alternatively, spin up a notebook (`notebooks/`) and load the checkpoint with `DistilBertCrfForTokenClassification.from_pretrained`.
 - 或直接运行：
   ```bash
   ./scripts/eval_baseline.sh
   ```

## 5. Monitoring & Troubleshooting
- Training runs emit JSON summaries to stdout; capture them for experiment tracking.
- Inspect `training_logs/*.log` for per-step losses, learning rates, and evaluation metrics.
- Append headline metrics to `results_summary.csv` after each run to keep report tables in sync.
- Early stopping triggers when validation F1 fails to improve for `patience` evaluations (see `configs/default.yaml`).
- For CPU-only environments, leave `fp16` enabled (no effect); AMP activates automatically when CUDA is available.
- Generate figures after successful runs:
  ```bash
  ../.venv/bin/python analysis/scripts/plot_metrics.py
  ../.venv/bin/python analysis/scripts/entity_stats.py
  ```
  Outputs land in `analysis/figures/` and CSV summaries accompany each plot.
- Feature toggles (`model.use_char_features`, `model.use_gazetteer`, `training.adversarial_training`, `training.lora`) currently raise `NotImplementedError`; leave them disabled until implementations are added.
- Baseline recap: see `docs/baseline_summary.md` for a concise table of metrics, figures, and error samples referenced in the final report.

## 6. Next Steps & Extensions
- Implement feature augmentations (char embeddings, gazetteer soft labels) and toggle them via new config fields.
- Integrate LoRA/adversarial training once the corresponding modules are added to `modeling.py` and `trainer.py`.
- Update `docs/implementation_notes.md` whenever new functions or scripts are introduced to keep documentation in lockstep with the codebase.
