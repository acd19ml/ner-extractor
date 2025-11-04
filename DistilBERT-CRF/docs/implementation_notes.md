# Implementation Notes

## src/config.py

### load_config
- **Purpose**: Read the YAML configuration and normalize entries under the `paths` section to absolute `Path` objects. This gives downstream code a consistent way to reason about filesystem locations regardless of where the CLI is executed.
- **Key Idea**: Resolve the configuration file location first, use `yaml.safe_load`, and lift every path entry relative to the directory containing the config. Mapping validation guards against empty configs.
- **Further Work**: Introduce schema validation (e.g. with `pydantic`) to detect missing keys early and surface richer error messages.

### save_config
- **Purpose**: Persist run-time configuration (e.g. after auto-tuning) while keeping relative paths readable inside the resulting YAML.
- **Key Idea**: Copy the mapping, convert any `Path` values back to relative strings when possible, and rely on `yaml.safe_dump` for serialization.
- **Further Work**: Support merging run metadata (timestamps, git commit) and optionally enforce atomic writes using temporary files.

## configs/default.yaml / configs/sanity.yaml

- **New Fields**: Added `use_char_features`, `char_hidden_size`, `use_gazetteer`, `gazetteer_weight` under `model`, and placeholder entries for `adversarial_training` plus a `lora` block under `training`.
- **Purpose**: Reserve configuration toggles for upcoming experiments (character CNNs, gazetteer priors, adversarial fine-tuning, LoRA) so future code can branch on explicit config values.
- **Further Work**: Once implementations land in `modeling.py`/`trainer.py`, enforce schema checks so unsupported combinations fail fast, and document expected value ranges.
## src/utils.py

### set_seed
- **Purpose**: Ensure reproducible experiments by seeding Python, NumPy, and PyTorch RNGs when available.
- **Key Idea**: Guard Torch usage behind an import check so the helper still works before dependencies are installed.
- **Further Work**: Extend to seed other frameworks (e.g. Hugging Face `transformers` and `random` components) once they are integrated.

### ensure_dirs
- **Purpose**: Create one or more directories declared in config files and return their absolute paths for later logging.
- **Key Idea**: Iterate over inputs, expand user symbols, enforce parent creation, and normalize to a dictionary keyed by the original string.
- **Further Work**: Surface whether a directory was newly created vs. already present, enabling richer user feedback.

### timestamped_run_dir
- **Purpose**: Allocate a unique, timestamped directory for experiment artifacts under a configurable root.
- **Key Idea**: Compose a canonical timestamp string and append a numeric suffix when collisions occur due to rapid successive calls.
- **Further Work**: Allow custom strftime templates and optionally inject run metadata (e.g. short git hash) into the directory name.

### create_logger
- **Purpose**: Provide a consistently formatted logger with both console and optional file sinks while avoiding duplicate handlers.
- **Key Idea**: Reuse logger instances by name, attach stream/file handlers only when not already present, and standardize message format.
- **Further Work**: Integrate JSON logging for structured log ingestion and expose handler configuration through the YAML config.

## src/datasets.py

### ConllSentence.from_lines
- **Purpose**: Transform raw CoNLL token lines into a typed container separating tokens, POS, chunk, and NER labels.
- **Key Idea**: Validate each line has four columns and push parsing responsibility away from higher-level routines.
- **Further Work**: Support optional extra columns (e.g. document IDs) without breaking compatibility.

### ConllSentence.to_lines
- **Purpose**: Serialize the in-memory representation back into CoNLL format for downstream tooling compatibility.
- **Key Idea**: Zip the four parallel lists and format them line-by-line, keeping BIO tags untouched.
- **Further Work**: Make whitespace strategy configurable (tabs vs spaces) and expose gzip compression for large corpora.

### ConllSentence.primary_label
- **Purpose**: Provide a single label representative of the sentence for stratified sampling.
- **Key Idea**: Return the first non-`O` entity suffix, falling back to `O` when a sentence has no entities.
- **Further Work**: Track full label distributions to enable multi-label aware sampling strategies.

### read_conll_file
- **Purpose**: Parse an on-disk CoNLL file into a list of `ConllSentence` objects, skipping document markers.
- **Key Idea**: Stream the file line-by-line, accumulate sentence buffers, and leverage `ConllSentence.from_lines` for validation.
- **Further Work**: Detect and warn about annotation inconsistencies (e.g. illegal BIO transitions) during ingestion.

### write_conll_file
- **Purpose**: Emit processed sentences back onto disk, preserving compatibility with common evaluation scripts.
- **Key Idea**: Ensure the destination directory exists, then write each sentence followed by a blank line separator.
- **Further Work**: Add checksum logging to help trace data lineage and detect accidental modification.

### stratified_split
- **Purpose**: Perform a train/validation split that approximately preserves entity distributions across splits.
- **Key Idea**: Use `scikit-learn`'s `train_test_split` with labels obtained from `ConllSentence.primary_label`.
- **Further Work**: Replace coarse single-label stratification with multi-label or length-aware sampling to handle edge cases.

### prepare_conll03_corpus
- **Purpose**: Build normalized train/validation/test files under `data/processed` for the project pipeline.
- **Key Idea**: Locate raw files under common CoNLL naming conventions, merge train/dev if available, stratify a new validation split, and persist all splits.
- **Further Work**: Cache intermediate metadata (e.g. entity histograms) and optionally expose K-fold splits for later cross-validation.

## src/tokenization.py

### load_tokenizer
- **Purpose**: Lazily load and cache Hugging Face tokenizers so repeated invocations (training, evaluation, notebooks) don't trigger redundant downloads.
- **Key Idea**: Wrap `AutoTokenizer.from_pretrained` with `functools.lru_cache`, enforce usage of fast tokenizers for word-alignment, and surface a clear error if a slow tokenizer is returned.
- **Further Work**: Allow injecting tokenizer kwargs (e.g. additional special tokens) while preserving cacheability.

### prepare_tokenizer
- **Purpose**: Provide a single helper to load and normalize tokenizer settings (max sequence length, truncation/padding sides) used across the pipeline.
- **Key Idea**: Delegate loading to `load_tokenizer`, optionally override `model_max_length`, and standardize truncation behavior for right-padded mini-batches.
- **Further Work**: Persist tokenizer config snapshots per experiment to make runs more reproducible and traceable.

## src/data_module.py

### collect_unique_labels
- **Purpose**: Build deterministic BIO label mappings from training sentences, enabling consistent ID assignments for CRF heads and metrics.
- **Key Idea**: Count labels across the corpus, sort at the end, and expose both `label_to_id` and its inverse via a small dataclass.
- **Further Work**: Extend to include frequency thresholds or merge rare labels when experimenting with noisy corpora.

### load_processed_conll
- **Purpose**: Read prepared train/validation/test splits from disk before tokenization.
- **Key Idea**: Enforce the expected filenames under `data/processed`, reuse `read_conll_file`, and raise a clean error if any split is missing.
- **Further Work**: Support lazy loading or memory-mapped datasets for extremely large corpora.

### TokenizedNERDataset
- **Purpose**: Convert `ConllSentence` objects into token-aligned tensors ready for Transformer training.
- **Key Idea**: Call the fast tokenizer with split tokens, leverage `word_ids` for BIO label alignment, and mark sub-token continuations with `-100` unless explicitly requested.
- **Further Work**: Cache tokenizer outputs to disk for repeated experiments or add dynamic truncation strategies based on sentence length distributions.

### create_dataloaders
- **Purpose**: Assemble PyTorch dataloaders for all splits while sharing a consistent data collator and label vocabulary.
- **Key Idea**: Load sentences, derive the label set from the union of train/validation/test splits (so limited train samples still cover evaluation tags), optionally trim to a fixed number of samples for quick sanity checks, instantiate the tokenized dataset per split, and wrap them with `DataCollatorForTokenClassification`.
- **Further Work**: Add support for distributed sampler integration and expose iterable-style dataloaders for streaming large corpora.

## src/modeling.py

### DistilBertCrfConfig
- **Purpose**: Collect model hyperparameters (pretrained checkpoint, label count, dropout settings, pad label id, feature toggles) in a simple object that aligns with the YAML config.
- **Key Idea**: Avoid polluting the Hugging Face config with experiment-specific values while still passing CRF/feature-related details (e.g., chars/gazetteer flags) to the model constructor.
- **Further Work**: Replace with a dataclass referencing the full HF config once we integrate LoRA/adversarial flags.

### DistilBertCrfForTokenClassification
- **Purpose**: Combine DistilBERT encoder with a linear emission head and a CRF layer for structured prediction.
- **Key Idea**: Reuse pretrained weights, apply dropout, sanitize labels by replacing `-100` with the configured pad id before passing them to TorchCRF, and decode sequences with `viterbi_decode` while respecting attention masks. Currently raises `NotImplementedError` when the new `use_char_features` or `use_gazetteer` flags are enabled to signal missing implementations.
- **Further Work**: Inject additional feature projections (char embeddings, gazetteer logits) via concatenation before the classifier.

### freeze_encoder_layers
- **Purpose**: Allow partial freezing of early DistilBERT layers during fine-tuning to improve stability or save compute.
- **Key Idea**: Iterate over the transformer layers and flip `requires_grad` based on an integer budget.
- **Further Work**: Expose more granular control (e.g. freeze attention vs feed-forward submodules separately).

### count_trainable_parameters
- **Purpose**: Provide quick diagnostics on parameter counts for logging and report tables.
- **Key Idea**: Sum `.numel()` over parameters and split by `requires_grad`.
- **Further Work**: Extend to report parameter groups (backbone vs CRF head) to contextualize PEFT experiments.

## src/metrics.py

### align_predictions
- **Purpose**: Convert CRF integer outputs and gold label IDs into BIO tag sequences while respecting `-100` ignore indices created during tokenization.
- **Key Idea**: Iterate token-wise across each sentence, drop masked positions, and map IDs to label strings via `id_to_label`.
- **Further Work**: Handle nested/overlapping entity schemes or convert to span tuples directly for richer analysis.

### compute_ner_metrics
- **Purpose**: Produce headline precision/recall/F1/accuracy scores using `seqeval`, aligning with common NER reporting standards.
- **Key Idea**: Reuse `align_predictions` to generate clean BIO sequences, then delegate to `seqeval` metric functions.
- **Further Work**: Surface micro/macro variants and per-entity breakdowns to match report table requirements.

### compute_entity_counts
- **Purpose**: Quickly summarise entity frequencies to feed exploratory analysis and charting scripts.
- **Key Idea**: Leverage `seqeval.get_entities` to extract entity spans and aggregate counts per entity type.
- **Further Work**: Track span lengths, boundary errors, and cross-compare predictions vs. references for confusion analysis.

## src/trainer.py

### TrainerConfig
- **Purpose**: Encapsulate high-level training hyperparameters (optimization, logging cadence, checkpointing) detached from the YAML config so they can be constructed programmatically.
- **Key Idea**: Use a dataclass with sensible defaults and filesystem paths, enabling overrides from CLI arguments or notebooks.
- **Further Work**: Mirror the Hugging Face `TrainingArguments` schema for easier interoperability with existing tooling.

### NerTrainer.__init__
- **Purpose**: Wire the model, dataloaders, and config together, preparing optimizer, scheduler, automatic mixed precision, and logging.
- **Key Idea**: Seed RNGs, send the model to the target device, instantiate optim/sched via helper methods, and prime bookkeeping (step counters, histories). Currently raises informative `NotImplementedError` if `adversarial_training` or `lora` configs are supplied before those paths are implemented.
- **Further Work**: Add support for distributed/Accelerate setups and configurable optimizer factories.

### NerTrainer.train
- **Purpose**: Execute the full fine-tuning loop with gradient accumulation, periodic evaluation, checkpointing, and patience-based early stopping.
- **Key Idea**: Iterate over epochs, wrap the forward pass in `torch.amp.autocast` when mixed precision is enabled, clip gradients, update scheduler per accumulation cycle, trigger evaluation/save hooks based on global steps.
- **Further Work**: Surface richer telemetry (tensorboard, wandb) and allow customizable callback hooks for advanced experiments.

### NerTrainer.evaluate
- **Purpose**: Run a forward-only pass on validation/test splits, aggregate CRF predictions, and compute seqeval metrics.
- **Key Idea**: Request decoded sequences during evaluation, map tensors back to CPU lists, and feed them into `compute_ner_metrics` while logging results.
- **Further Work**: Return per-label metrics, confusion matrices, and optional span-level audits for report-ready tables.

### NerTrainer._save_checkpoint
- **Purpose**: Persist model weights, label mappings, and trainer configuration at checkpoints (best model, periodic snapshots).
- **Key Idea**: Use `model.save_pretrained`, dump the label map to JSON, and store config fields for reproducibility; support overwrite semantics for the "best" tag.
- **Further Work**: Atomically swap checkpoints, store optimizer/scheduler states for resuming, and integrate experiment metadata (git hash, tokenizer config).

### NerTrainer._handle_evaluation
- **Purpose**: Compare evaluation F1 against the running best, manage patience counters, and trigger checkpoint saves when improvements occur.
- **Key Idea**: Append metric snapshots to history, reset patience on improvements, otherwise increment until early stopping kicks in.
- **Further Work**: Allow custom early-stopping metrics (e.g. recall) or multi-metric composite criteria.

### NerTrainer._setup_optimizer/_setup_scheduler/_max_train_steps
- **Purpose**: Provide reusable helpers that configure the AdamW optimizer with weight decay separation, compute total training steps, and instantiate a linear warmup/decay scheduler.
- **Key Idea**: Mirror Hugging Face best practices while keeping the implementation lightweight and framework-agnostic.
- **Further Work**: Expose alternative schedulers (cosine, constant) and optimizer families (Lion, Adafactor) via configuration.

## scripts/prepare_conll03.py

### build_arg_parser
- **Purpose**: Define CLI arguments enabling configuration-driven or ad-hoc execution of the dataset preparation flow.
- **Key Idea**: Provide overrides for key parameters (`raw_dir`, `processed_dir`, `val_ratio`, `seed`) while defaulting to YAML entries.
- **Further Work**: Add boolean flags for gazetteer generation and augmentation hooks once those components are available.

### main
- **Purpose**: Wire argument parsing, configuration loading, and corpus preparation into an executable script.
- **Key Idea**: Resolve overrides in priority order (CLI > config), execute the preparation routine, and print resulting file paths.
- **Further Work**: Return structured exit codes and integrate with richer logging once the training CLI is in place.

## scripts/train_distilbert_crf.py

### build_arg_parser
- **Purpose**: Offer a CLI for configuring training runs (config path, run naming, layer freezing, sample limiting, skip-training evaluation, test evaluation toggle).
- **Key Idea**: Provide ergonomic defaults mapped to the YAML file while exposing optional knobs that align with experimental needs, including quick-sanity constraints and reuse of existing checkpoints.
- **Further Work**: Add flags for LoRA/adversarial toggles once those components land in the model/trainer.

### main
- **Purpose**: Orchestrate tokenization, dataloader construction, model instantiation, trainer configuration, and the full training/eval lifecycle.
- **Key Idea**: Read YAML config, resolve local directories when `pretrained_model_name` points to disk, build dependent objects (tokenizer, dataloaders, model, trainer), normalise sample limits (`<=0` → full dataset), optionally load pre-existing checkpoints (supports `pytorch_model.bin` or `model.safetensors`) when `--skip-training` is set, run training, and emit JSON summaries for downstream logging.
- **Further Work**: Persist full run metadata (metrics history, config snapshot), surface CLI overrides for scheduler/optimizer choices, and integrate with experiment tracking services.

## results_summary.csv

- **Purpose**: Tabulate headline metrics for each experiment (`run_name`, validation/test precision/recall/F1, optional loss & accuracy) to drive report tables and comparisons.
- **Current Entries**: `distilbert_crf_full` validation F1≈0.9438, test F1≈0.8959 (precision≈0.892, recall≈0.900, accuracy≈0.9794, loss≈1.82).
- **Further Work**: Append new rows for ablation variants and Kaggle/benchmark submissions; include seed/config hashes for reproducibility.

## analysis/scripts/plot_metrics.py

- **Purpose**: Parse `training_logs/distilbert_crf_full.log` and emit training loss and validation metric curves alongside CSV tables for reproducible plotting.
- **Key Idea**: Use regex to extract training steps/loss and validation metrics, dump Pandas DataFrames, and render Matplotlib figures to `analysis/figures/`.
- **Further Work**: Overlay multiple runs for comparison and include moving-average smoothing or learning-rate plots.

## analysis/scripts/entity_stats.py

- **Purpose**: Summarise dataset characteristics (sentence lengths, entity frequency) with plots and tabular exports.
- **Key Idea**: Reuse the existing CoNLL loader, aggregate counts via `Counter`, and generate box/bar plots saved in `analysis/figures/`.
- **Further Work**: Extend to per-split breakdowns, rare-entity exploration, and export representative samples for error analysis.

## docs/baseline_summary.md

- **Purpose**: Snapshot the baseline DistilBERT-CRF experiment (commands, metrics, figures, error cases) for quick reference in reports.
- **Further Work**: Refresh after each milestone to contrast improvements (append tables or links to additional figures).
- **Supplementary Scripts**: `scripts/train_baseline.sh` and `scripts/eval_baseline.sh` wrap the main CLI for quick baseline reproduction (train + test evaluate).
- **Baseline Labels**: `data/labels.json` stores the BIO tag vocabulary extracted from the processed training split for downstream tooling.
