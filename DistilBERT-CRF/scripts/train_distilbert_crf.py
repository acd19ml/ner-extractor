"""Command-line entrypoint for training the DistilBERT-CRF NER model."""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import load_config  # noqa: E402
from data_module import create_dataloaders  # noqa: E402
from modeling import (  # noqa: E402
    DistilBertCrfConfig,
    DistilBertCrfForTokenClassification,
    freeze_encoder_layers,
)
from tokenization import prepare_tokenizer  # noqa: E402
from trainer import NerTrainer, TrainerConfig  # noqa: E402
from utils import ensure_dirs  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT_DIR / "configs" / "default.yaml"),
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="distilbert_crf",
        help="Name for this training run (affects logs and checkpoints).",
    )
    parser.add_argument(
        "--freeze-encoder-layers",
        type=int,
        default=0,
        help="Number of initial DistilBERT layers to freeze.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the best checkpoint on the test split after training.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit the number of training sentences used (sanity checks).",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit the number of validation/test sentences used.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip fine-tuning and only run evaluation using an existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="best",
        help="Checkpoint subdirectory to load when skipping training (default: best).",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default=None,
        help="Optional path to write JSON metrics summary for this run.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    optimizer_cfg = config.get("optimizer", {})
    regularization_cfg = config.get("regularization", {})
    evaluation_cfg = config["evaluation"]

    ensure_dirs(
        [
            config["paths"]["processed_data_dir"],
            config["paths"]["models_dir"],
            config["paths"]["logs_dir"],
        ]
    )

    pretrained_model_name = model_cfg["pretrained_model_name"]
    candidate_roots = [
        Path(pretrained_model_name).expanduser(),
        Path(ROOT_DIR) / pretrained_model_name,
        Path(config["paths"]["models_dir"]).parent / pretrained_model_name,
    ]
    for candidate in candidate_roots:
        expanded = candidate.expanduser()
        if expanded.exists():
            pretrained_model_name = str(expanded.resolve())
            break

    tokenizer = prepare_tokenizer(
        pretrained_model_name=pretrained_model_name,
        max_length=dataset_cfg["max_seq_length"],
    )

    max_train_samples = args.max_train_samples
    max_eval_samples = args.max_eval_samples
    if max_train_samples is not None and max_train_samples <= 0:
        max_train_samples = None
    if max_eval_samples is not None and max_eval_samples <= 0:
        max_eval_samples = None

    fold_indices = _load_fold_indices_from_env()
    fold_index = _get_fold_index()

    dataloaders, label_info = create_dataloaders(
        processed_dir=config["paths"]["processed_data_dir"],
        tokenizer=tokenizer,
        max_length=dataset_cfg["max_seq_length"],
        batch_size=training_cfg["batch_size"],
        eval_batch_size=evaluation_cfg["eval_batch_size"],
        label_all_tokens=False,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        augmentation_cfg=config.get("augmentation"),
        seed=training_cfg["seed"],
        fold_indices=fold_indices,
    )

    dropout_emission = model_cfg.get("dropout_emission", model_cfg.get("dropout", 0.1))
    distilbert_config = DistilBertCrfConfig(
        pretrained_model_name=pretrained_model_name,
        num_labels=len(label_info.labels),
        dropout=dropout_emission,
        crf_dropout=model_cfg.get("crf_dropout", 0.0),
        pad_label_id=label_info.label_to_id.get("O", 0),
        use_char_features=model_cfg.get("use_char_features", False),
        char_hidden_size=model_cfg.get("char_hidden_size"),
        use_gazetteer=model_cfg.get("use_gazetteer", False),
        gazetteer_weight=model_cfg.get("gazetteer_weight"),
    )

    model = DistilBertCrfForTokenClassification(distilbert_config)
    if args.freeze_encoder_layers > 0:
        freeze_encoder_layers(model, args.freeze_encoder_layers)

    adversarial_cfg = training_cfg.get("adversarial_training") or None
    lora_cfg = training_cfg.get("lora")
    if isinstance(lora_cfg, dict) and not lora_cfg.get("enabled", False):
        lora_cfg = None

    freeze_cfg = training_cfg.get("freeze", {})
    freeze_initial_layers = freeze_cfg.get("initial_layers", 0)
    freeze_min_layers = freeze_cfg.get("min_frozen_layers", freeze_cfg.get("min_trainable_layer", 0))
    unfreeze_every = freeze_cfg.get("unfreeze_every_n_epochs", 0)

    trainer_config = TrainerConfig(
        num_epochs=training_cfg["num_epochs"],
        learning_rate=optimizer_cfg.get("encoder_lr", training_cfg.get("learning_rate", 3e-5)),
        encoder_learning_rate=optimizer_cfg.get("encoder_lr"),
        head_learning_rate=optimizer_cfg.get("head_lr"),
        weight_decay=optimizer_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.01)),
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        max_grad_norm=optimizer_cfg.get("max_grad_norm", training_cfg.get("max_grad_norm", 1.0)),
        warmup_ratio=optimizer_cfg.get("warmup_ratio", training_cfg.get("warmup_ratio", 0.1)),
        scheduler_type=optimizer_cfg.get("scheduler", "linear"),
        llrd_gamma=optimizer_cfg.get("llrd_gamma", 1.0),
        fp16=training_cfg["fp16"],
        logging_steps=training_cfg["logging_steps"],
        eval_steps=training_cfg["eval_steps"],
        save_steps=training_cfg["save_steps"],
        patience=training_cfg["patience"],
        seed=training_cfg["seed"],
        output_dir=Path(config["paths"]["models_dir"]) / args.run_name,
        logs_dir=Path(config["paths"]["logs_dir"]),
        run_name=args.run_name,
        adversarial_training=adversarial_cfg,
        lora_config=lora_cfg,
        rdrop_alpha=regularization_cfg.get("rdrop_lambda", 0.0),
        use_ema=regularization_cfg.get("use_ema", False),
        ema_decay=regularization_cfg.get("ema_decay", 0.999),
        crf_l2=regularization_cfg.get("crf_l2", 0.0),
        freeze_initial_layers=freeze_initial_layers,
        unfreeze_every_n_epochs=unfreeze_every,
        freeze_min_layers=freeze_min_layers,
    )

    trainer = NerTrainer(
        model=model,
        dataloaders=dataloaders,
        label_mapping=label_info.__dict__,
        config=trainer_config,
    )

    if args.skip_training:
        checkpoint_dir = Path(trainer_config.output_dir) / args.checkpoint_tag
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory '{checkpoint_dir}' not found. Run training first "
                "or specify a different tag via --checkpoint-tag."
            )
        state_dict_path = checkpoint_dir / "pytorch_model.bin"
        safetensors_path = checkpoint_dir / "model.safetensors"
        if state_dict_path.exists():
            loaded_path = state_dict_path
            state_dict = torch.load(state_dict_path, map_location=trainer.device)
        elif safetensors_path.exists():
            loaded_path = safetensors_path
            state_dict = load_safetensors(safetensors_path)
        else:
            raise FileNotFoundError(
                f"Missing model weights at {state_dict_path} or {safetensors_path}. Ensure the checkpoint is valid."
            )
        trainer.model.load_state_dict(state_dict, strict=False)
        trainer.logger.info("Loaded checkpoint from %s", loaded_path)

        eval_split = "test" if args.evaluate_test else "validation"
        eval_metrics = trainer.evaluate(split=eval_split)
        print(f"{eval_split.capitalize()} metrics:")
        print(json.dumps(eval_metrics, indent=2))

        summary = {
            "run_name": args.run_name,
            "mode": "evaluation",
            "fold_index": fold_index,
            "checkpoint_tag": args.checkpoint_tag,
            "evaluation_split": eval_split,
            "evaluation_metrics": eval_metrics,
        }
        _write_metrics_output(summary, args.metrics_output)
        return

    train_summary = trainer.train()

    print("Training completed.")
    print(json.dumps({"best_f1": train_summary["best_f1"], "model_dir": str(train_summary["best_model_dir"])}, indent=2))

    summary: Dict[str, Any] = {
        "run_name": args.run_name,
        "fold_index": fold_index,
        "best_f1": train_summary["best_f1"],
        "best_model_dir": str(train_summary["best_model_dir"]),
    }

    if args.evaluate_test:
        test_metrics = trainer.evaluate(split="test")
        print("Test set metrics:")
        print(json.dumps(test_metrics, indent=2))
        summary["test_metrics"] = test_metrics

    _write_metrics_output(summary, args.metrics_output)


def _load_fold_indices_from_env() -> Optional[dict]:
    """Parse fold selection indices from environment variables if provided."""

    train_env = os.environ.get("NER_FOLD_TRAIN_INDICES")
    val_env = os.environ.get("NER_FOLD_VALIDATION_INDICES")
    if not train_env or not val_env:
        return None

    try:
        train_indices = json.loads(train_env)
        val_indices = json.loads(val_env)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in NER_FOLD_* environment variables.") from exc

    if not isinstance(train_indices, list) or not isinstance(val_indices, list):
        raise ValueError("Fold indices must decode to lists of integers.")

    return {"train": [int(idx) for idx in train_indices], "validation": [int(idx) for idx in val_indices]}


def _get_fold_index() -> Optional[int]:
    value = os.environ.get("NER_FOLD_INDEX")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _write_metrics_output(summary: Dict[str, Any], output_path: Optional[str]) -> None:
    if not output_path:
        return
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
