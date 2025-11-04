"""Training loop implementation for DistilBERT-CRF NER."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup

from metrics import MetricsResult, compute_ner_metrics
from modeling import DistilBertCrfForTokenClassification, count_trainable_parameters
from utils import create_logger, ensure_dirs, set_seed


@dataclass
class TrainerConfig:
    """Configuration values for the training loop."""

    num_epochs: int = 5
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    fp16: bool = True
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 400
    patience: int = 5
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("models/distilbert_crf"))
    logs_dir: Path = field(default_factory=lambda: Path("training_logs"))
    run_name: str = "distilbert_crf"
    adversarial_training: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None


class NerTrainer:
    """High-level API to train and evaluate the DistilBERT-CRF model."""

    def __init__(
        self,
        model: DistilBertCrfForTokenClassification,
        dataloaders: Mapping[str, DataLoader],
        label_mapping: Mapping[str, Any],
        config: TrainerConfig,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.dataloaders = dataloaders
        self.label_mapping = label_mapping  # expects keys: labels, label_to_id, id_to_label
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_seed(config.seed)
        self.model.to(self.device)

        ensure_dirs(
            [
                self.config.output_dir,
                self.config.logs_dir,
            ]
        )

        log_path = Path(self.config.logs_dir) / f"{self.config.run_name}.log"
        self.logger = create_logger("trainer", log_file=log_path)

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler(total_steps=self._max_train_steps())

        if self.config.adversarial_training:
            raise NotImplementedError(
                "Adversarial training has not been implemented yet. "
                "Please set `training.adversarial_training` to null."
            )
        if self.config.lora_config:
            raise NotImplementedError(
                "LoRA fine-tuning has not been implemented yet. "
                "Please set `training.lora.enabled` to false."
            )

        self.scaler: Optional[torch.cuda.amp.GradScaler]
        if self.config.fp16 and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.global_step = 0
        self.best_f1 = -math.inf
        self.patience_counter = 0
        self.train_history: List[Dict[str, Any]] = []
        self.eval_history: List[Dict[str, Any]] = []

        trainable_params, total_params = count_trainable_parameters(self.model)
        self.logger.info(
            "Model initialized. trainable_params=%s total_params=%s device=%s",
            trainable_params,
            total_params,
            self.device,
        )

    def _max_train_steps(self) -> int:
        train_loader = self.dataloaders["train"]
        updates_per_epoch = math.ceil(len(train_loader))
        return math.ceil(
            (updates_per_epoch * self.config.num_epochs) / self.config.gradient_accumulation_steps
        )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        return optimizer

    def _setup_scheduler(self, total_steps: int):
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def train(self) -> Dict[str, Any]:
        """Run the training loop with evaluation and early stopping."""

        train_loader = self.dataloaders["train"]
        gradient_accumulation = self.config.gradient_accumulation_steps

        for epoch in range(1, self.config.num_epochs + 1):
            self.logger.info("Epoch %s/%s", epoch, self.config.num_epochs)
            epoch_loss = 0.0
            step_loss = 0.0

            self.model.train()
            self.optimizer.zero_grad()

            total_steps = len(train_loader)
            for step, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                batch.pop("sentence_index", None)

                autocast_context = (
                    torch.amp.autocast("cuda", enabled=True)
                    if self.scaler is not None
                    else nullcontext()
                )
                with autocast_context:
                    outputs = self.model(labels=labels, return_predictions=False, **batch)
                    loss = outputs.loss

                if loss is None:
                    raise ValueError("Model did not return a loss during training.")

                loss = loss / gradient_accumulation

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                step_loss += loss.item()

                accumulation_boundary = step % gradient_accumulation == 0 or step == total_steps
                if accumulation_boundary:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    epoch_loss += step_loss
                    step_loss = 0.0

                    if self.global_step % self.config.logging_steps == 0:
                        self.logger.info(
                            "step=%s loss=%.5f lr=%.6f",
                            self.global_step,
                            epoch_loss / max(1, self.global_step),
                            self.scheduler.get_last_lr()[0],
                        )

                    if self.global_step % self.config.eval_steps == 0:
                        eval_metrics = self.evaluate(split="validation")
                        self._handle_evaluation(eval_metrics)

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(tag=f"step_{self.global_step}")

            # Evaluate at epoch end
            eval_metrics = self.evaluate(split="validation")
            self._handle_evaluation(eval_metrics, epoch=epoch)

            if self.patience_counter >= self.config.patience:
                self.logger.info("Early stopping triggered.")
                break

        best_path = Path(self.config.output_dir) / "best"
        return {
            "best_f1": self.best_f1,
            "best_model_dir": best_path,
            "train_history": self.train_history,
            "eval_history": self.eval_history,
        }

    def evaluate(self, split: str = "validation") -> Dict[str, Any]:
        """Run evaluation on the specified split."""

        self.model.eval()
        dataloader = self.dataloaders[split]

        preds: List[List[int]] = []
        refs: List[List[int]] = []
        losses: List[float] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                batch.pop("sentence_index", None)

                outputs = self.model(labels=labels, return_predictions=True, **batch)
                if outputs.loss is not None:
                    losses.append(outputs.loss.item())

                if outputs.predictions is None:
                    raise ValueError("Model did not return predictions during evaluation.")

                preds.extend(outputs.predictions)
                refs.extend(labels.cpu().tolist())

        metrics: MetricsResult = compute_ner_metrics(preds, refs, self.label_mapping["id_to_label"])
        result = {
            "split": split,
            "loss": float(sum(losses) / max(1, len(losses))),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "accuracy": metrics.accuracy,
            "global_step": self.global_step,
        }
        self.eval_history.append(result)
        self.logger.info(
            "%s metrics | loss=%.5f precision=%.4f recall=%.4f f1=%.4f accuracy=%.4f",
            split,
            result["loss"],
            result["precision"],
            result["recall"],
            result["f1"],
            result["accuracy"],
        )
        return result

    def _handle_evaluation(self, eval_metrics: Dict[str, Any], epoch: Optional[int] = None) -> None:
        f1 = eval_metrics["f1"]
        self.train_history.append(
            {
                "step": self.global_step,
                "epoch": epoch,
                "f1": f1,
                "precision": eval_metrics["precision"],
                "recall": eval_metrics["recall"],
                "loss": eval_metrics["loss"],
            }
        )

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.patience_counter = 0
            self.logger.info("New best F1 %.4f achieved. Saving model.", f1)
            self._save_checkpoint(tag="best", overwrite=True)
        else:
            self.patience_counter += 1
            self.logger.info("No improvement. patience=%s/%s", self.patience_counter, self.config.patience)

    def _save_checkpoint(self, tag: str, overwrite: bool = False) -> Path:
        checkpoint_dir = Path(self.config.output_dir) / tag
        if checkpoint_dir.exists() and overwrite:
            for file in checkpoint_dir.iterdir():
                if file.is_file():
                    file.unlink()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        label_file = checkpoint_dir / "label_map.json"
        if not label_file.exists() or overwrite:
            import json

            with label_file.open("w", encoding="utf-8") as handle:
                json.dump(self.label_mapping["label_to_id"], handle, indent=2)

        torch.save(self.config.__dict__, checkpoint_dir / "trainer_config.pt")
        return checkpoint_dir
