"""Command-line entrypoint for training the DistilBERT-CRF NER model."""

from __future__ import annotations

import argparse
import json
import sys
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
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
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

    dataloaders, label_info = create_dataloaders(
        processed_dir=config["paths"]["processed_data_dir"],
        tokenizer=tokenizer,
        max_length=dataset_cfg["max_seq_length"],
        batch_size=training_cfg["batch_size"],
        eval_batch_size=evaluation_cfg["eval_batch_size"],
        label_all_tokens=False,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )

    distilbert_config = DistilBertCrfConfig(
        pretrained_model_name=pretrained_model_name,
        num_labels=len(label_info.labels),
        dropout=model_cfg["dropout"],
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

    trainer_config = TrainerConfig(
        num_epochs=training_cfg["num_epochs"],
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        max_grad_norm=training_cfg["max_grad_norm"],
        warmup_ratio=training_cfg["warmup_ratio"],
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

        if args.evaluate_test:
            test_metrics = trainer.evaluate(split="test")
            print("Test set metrics:")
            print(json.dumps(test_metrics, indent=2))
        else:
            val_metrics = trainer.evaluate(split="validation")
            print("Validation metrics:")
            print(json.dumps(val_metrics, indent=2))
        return

    train_summary = trainer.train()

    print("Training completed.")
    print(json.dumps({"best_f1": train_summary["best_f1"], "model_dir": str(train_summary["best_model_dir"])}, indent=2))

    if args.evaluate_test:
        test_metrics = trainer.evaluate(split="test")
        print("Test set metrics:")
        print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
