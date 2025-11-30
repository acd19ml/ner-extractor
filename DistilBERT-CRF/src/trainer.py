"""Training loop implementation for DistilBERT-CRF NER."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

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
    scheduler_type: str = "linear"
    encoder_learning_rate: Optional[float] = None
    head_learning_rate: Optional[float] = None
    llrd_gamma: float = 1.0
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
    rdrop_alpha: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.999
    crf_l2: float = 0.0
    freeze_initial_layers: int = 0
    unfreeze_every_n_epochs: int = 0
    freeze_min_layers: int = 0


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
        self.device = device or self._auto_device()

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
        encoder_layers = getattr(self.model.distilbert, "transformer", None)
        layer_list = getattr(encoder_layers, "layer", [])
        self.total_encoder_layers = len(layer_list)
        self.unfreeze_interval = max(0, self.config.unfreeze_every_n_epochs)
        self.freeze_min_layers = min(
            max(0, self.config.freeze_min_layers),
            self.total_encoder_layers,
        )
        self.currently_frozen_layers = 0
        if self.config.freeze_initial_layers > 0:
            self._set_encoder_trainability(self.config.freeze_initial_layers)

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
        self.use_ema = bool(self.config.use_ema)
        self.ema_state: Dict[str, torch.Tensor] = {}
        if self.use_ema:
            self._initialize_ema_state()

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
        """Create AdamW parameter groups with differential LR + LLRD support."""

        base_encoder_lr = self.config.encoder_learning_rate or self.config.learning_rate
        head_lr = self.config.head_learning_rate or base_encoder_lr
        llrd_gamma = self.config.llrd_gamma or 1.0

        optimizer_groups: List[Dict[str, Any]] = []
        added_params: set[int] = set()
        no_decay_markers = ("bias", "LayerNorm", "layer_norm")

        def add_group(named_params: Iterable[Tuple[str, torch.nn.Parameter]], lr: float) -> None:
            decay_params: List[torch.nn.Parameter] = []
            no_decay_params: List[torch.nn.Parameter] = []
            for name, param in named_params:
                if not param.requires_grad:
                    continue
                param_id = id(param)
                if param_id in added_params:
                    continue
                added_params.add(param_id)
                target = (
                    no_decay_params
                    if any(marker in name for marker in no_decay_markers)
                    else decay_params
                )
                target.append(param)

            if decay_params:
                optimizer_groups.append(
                    {"params": decay_params, "weight_decay": self.config.weight_decay, "lr": lr}
                )
            if no_decay_params:
                optimizer_groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": lr})

        # Head: classifier + CRF transitions
        add_group(self.model.classifier.named_parameters(), head_lr)
        add_group(self.model.crf.named_parameters(), head_lr)

        # Encoder layers with layer-wise LR decay
        transformer_layers = list(self.model.distilbert.transformer.layer)
        for depth, layer in enumerate(reversed(transformer_layers)):
            layer_lr = base_encoder_lr * (llrd_gamma ** depth)
            add_group(layer.named_parameters(), layer_lr)

        # Embeddings (deepest portion of encoder)
        if hasattr(self.model.distilbert, "embeddings"):
            depth = len(transformer_layers)
            embeddings_lr = base_encoder_lr * (llrd_gamma ** depth)
            add_group(self.model.distilbert.embeddings.named_parameters(), embeddings_lr)

        # Any remaining trainable parameters
        leftover: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) not in added_params:
                leftover.append((name, param))
        if leftover:
            add_group(leftover, base_encoder_lr)

        self.logger.info(
            "Optimizer configured with %s groups | encoder_lr=%s head_lr=%s llrd_gamma=%s",
            len(optimizer_groups),
            base_encoder_lr,
            head_lr,
            llrd_gamma,
        )
        return AdamW(optimizer_groups, lr=base_encoder_lr)

    def _setup_scheduler(self, total_steps: int):
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler_type = (self.config.scheduler_type or "linear").lower()
        if scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
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
                sample_weight = batch.pop("loss_weight", None)
                batch.pop("sentence_index", None)

                autocast_context = (
                    torch.amp.autocast("cuda", enabled=True)
                    if self.scaler is not None
                    else nullcontext()
                )
                with autocast_context:
                    if self.config.rdrop_alpha > 0.0:
                        outputs_a = self.model(
                            labels=labels,
                            sample_weight=sample_weight,
                            return_predictions=False,
                            **batch,
                        )
                        outputs_b = self.model(
                            labels=labels,
                            sample_weight=sample_weight,
                            return_predictions=False,
                            **batch,
                        )
                        if outputs_a.loss is None or outputs_b.loss is None:
                            raise ValueError("Model did not return a loss during training.")
                        loss = (outputs_a.loss + outputs_b.loss) * 0.5
                        logits_a = outputs_a.logits
                        logits_b = outputs_b.logits
                        if logits_a is None or logits_b is None:
                            raise ValueError("Model did not return logits required for R-Drop.")
                        mask = batch.get("attention_mask")
                        if mask is None:
                            mask = labels.ne(-100)
                        rdrop_loss = self._compute_rdrop_loss(logits_a, logits_b, mask)
                        loss = loss + self.config.rdrop_alpha * rdrop_loss
                    else:
                        outputs = self.model(
                            labels=labels,
                            sample_weight=sample_weight,
                            return_predictions=False,
                            **batch,
                        )
                        loss = outputs.loss

                if loss is None:
                    raise ValueError("Model did not return a loss during training.")

                loss = self._apply_crf_regularization(loss)
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
                    self._update_ema()
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

            self._maybe_unfreeze_layers(epoch)

        best_path = Path(self.config.output_dir) / "best"
        return {
            "best_f1": self.best_f1,
            "best_model_dir": best_path,
            "train_history": self.train_history,
            "eval_history": self.eval_history,
        }

    def evaluate(self, split: str = "validation") -> Dict[str, Any]:
        """Run evaluation on the specified split."""

        dataloader = self.dataloaders[split]

        preds: List[List[int]] = []
        refs: List[List[int]] = []
        losses: List[float] = []

        with self._ema_weights_context():
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch.pop("labels")
                    batch.pop("loss_weight", None)
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
            if self.use_ema:
                self._save_checkpoint(tag="ema_best", overwrite=True, use_ema_weights=True)
        else:
            self.patience_counter += 1
            self.logger.info("No improvement. patience=%s/%s", self.patience_counter, self.config.patience)

    def _save_checkpoint(self, tag: str, overwrite: bool = False, use_ema_weights: bool = False) -> Path:
        with (self._ema_weights_context() if use_ema_weights else nullcontext()):
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

    def _auto_device(self) -> torch.device:
        """Select the best available device (CUDA → MPS → CPU)."""

        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _compute_rdrop_loss(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return symmetric KL divergence for R-Drop regularization."""

        if mask is None:
            mask = torch.ones_like(logits_a[..., 0], dtype=torch.bool)
        mask = mask.bool()
        log_prob_a = F.log_softmax(logits_a, dim=-1)
        log_prob_b = F.log_softmax(logits_b, dim=-1)
        prob_a = log_prob_a.exp()
        prob_b = log_prob_b.exp()

        kl_ab = F.kl_div(log_prob_a, prob_b, reduction="none").sum(dim=-1)
        kl_ba = F.kl_div(log_prob_b, prob_a, reduction="none").sum(dim=-1)
        sym_kl = 0.5 * (kl_ab + kl_ba)

        mask = mask.to(sym_kl.dtype)
        normalizer = mask.sum().clamp_min(1.0)
        return (sym_kl * mask).sum() / normalizer

    def _apply_crf_regularization(self, loss: torch.Tensor) -> torch.Tensor:
        """Add L2 penalty on CRF transition matrices if configured."""

        if self.config.crf_l2 <= 0:
            return loss
        crf_module = getattr(self.model, "crf", None)
        if crf_module is None:
            return loss
        penalty_terms: List[torch.Tensor] = []
        for attr in ("start_transitions", "end_transitions", "transitions"):
            param = getattr(crf_module, attr, None)
            if param is not None:
                penalty_terms.append(param.pow(2).sum())
        if not penalty_terms:
            return loss
        penalty = torch.stack(penalty_terms).sum()
        return loss + self.config.crf_l2 * penalty

    def _maybe_unfreeze_layers(self, epoch: int) -> None:
        """Gradually unfreeze encoder layers based on the configured schedule."""

        if self.unfreeze_interval <= 0:
            return
        if self.currently_frozen_layers <= self.freeze_min_layers:
            return
        if epoch % self.unfreeze_interval != 0:
            return
        previous_frozen = self.currently_frozen_layers
        new_frozen = max(self.freeze_min_layers, previous_frozen - 1)
        if new_frozen == previous_frozen:
            return
        self._set_encoder_trainability(new_frozen)
        unfrozen_layer = new_frozen
        self.logger.info(
            "Gradual unfreeze: epoch=%s unfroze_layer_index=%s frozen_layers=%s",
            epoch,
            unfrozen_layer,
            self.currently_frozen_layers,
        )

    def _set_encoder_trainability(self, frozen_layers: int) -> None:
        """Freeze the lowest ``frozen_layers`` encoder blocks."""

        encoder = getattr(self.model.distilbert, "transformer", None)
        layer_list = getattr(encoder, "layer", None)
        if layer_list is None:
            return
        total_layers = len(layer_list)
        freeze_upto = max(0, min(frozen_layers, total_layers))
        for idx, layer in enumerate(layer_list):
            trainable = idx >= freeze_upto
            for param in layer.parameters():
                param.requires_grad = trainable
        self.currently_frozen_layers = freeze_upto
        if freeze_upto > 0:
            self.logger.info("Encoder freeze applied: frozen_layers=%s/%s", freeze_upto, total_layers)
        else:
            self.logger.info("Encoder fully trainable (no frozen layers).")

    def _initialize_ema_state(self) -> None:
        """Snapshot initial parameters for EMA tracking."""

        self.ema_state = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.ema_state[name] = param.detach().clone()

    def _update_ema(self) -> None:
        """Update EMA weights after each optimizer step."""

        if not self.use_ema:
            return
        if not self.ema_state:
            self._initialize_ema_state()
            return
        decay = self.config.ema_decay
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                ema_tensor = self.ema_state.get(name)
                if ema_tensor is None:
                    self.ema_state[name] = param.detach().clone()
                    continue
                ema_tensor.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_weights_context(self):
        """Temporarily swap model weights to EMA values for eval/checkpointing."""

        if not self.use_ema or not self.ema_state:
            yield
            return

        backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                ema_tensor = self.ema_state.get(name)
                if ema_tensor is None:
                    continue
                backups.append((param, param.data.detach().clone()))
                param.data.copy_(ema_tensor)
        try:
            yield
        finally:
            with torch.no_grad():
                for param, stored in backups:
                    param.data.copy_(stored)
