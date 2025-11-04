"""Model definitions for DistilBERT-CRF NER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

try:  # TorchCRF uses a capitalized package name on some platforms
    from torchcrf import CRF
except ModuleNotFoundError:  # pragma: no cover - fallback for alternative packaging
    from TorchCRF import CRF  # type: ignore
from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


@dataclass
class ModelOutput(TokenClassifierOutput):
    """Custom output structure that adds CRF log-likelihood if available."""

    crf_log_likelihood: Optional[torch.Tensor] = None
    predictions: Optional[List[List[int]]] = None


class DistilBertCrfConfig:
    """Lightweight configuration container for DistilBERT+CRF models."""

    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        crf_dropout: float = 0.0,
        pad_label_id: int = 0,
        use_char_features: bool = False,
        char_hidden_size: Optional[int] = None,
        use_gazetteer: bool = False,
        gazetteer_weight: Optional[float] = None,
    ) -> None:
        self.pretrained_model_name = pretrained_model_name
        self.num_labels = num_labels
        self.dropout = dropout
        self.crf_dropout = crf_dropout
        self.pad_label_id = pad_label_id
        self.use_char_features = use_char_features
        self.char_hidden_size = char_hidden_size
        self.use_gazetteer = use_gazetteer
        self.gazetteer_weight = gazetteer_weight


class DistilBertCrfForTokenClassification(DistilBertPreTrainedModel):
    """DistilBERT encoder with linear projection and CRF decoding."""

    def __init__(self, config: DistilBertCrfConfig) -> None:
        base_config = DistilBertModel.from_pretrained(
            config.pretrained_model_name
        ).config  # reuse DistilBERT configuration
        base_config.num_labels = config.num_labels
        super().__init__(base_config)

        self.distilbert = DistilBertModel.from_pretrained(config.pretrained_model_name, config=base_config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(base_config.dim, config.num_labels)
        self.pad_label_id = config.pad_label_id
        self.crf = CRF(config.num_labels, pad_idx=None, use_gpu=torch.cuda.is_available())
        self.crf_dropout = nn.Dropout(config.crf_dropout)

        if config.use_char_features:
            raise NotImplementedError(
                "Character-level features are not implemented yet. "
                "Disable `model.use_char_features` or add the char module."
            )
        if config.use_gazetteer:
            raise NotImplementedError(
                "Gazetteer fusion is not implemented yet. "
                "Disable `model.use_gazetteer` or provide the gazetteer integration."
            )

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_predictions: bool = False,
    ) -> ModelOutput:
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        loss: Optional[torch.Tensor] = None
        crf_log_likelihood: Optional[torch.Tensor] = None
        predictions: Optional[List[List[int]]] = None

        mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            mask = attention_mask.bool()

        if labels is not None:
            if mask is None:
                mask = labels.ne(-100)
            labels_for_crf = labels.clone()
            labels_for_crf = labels_for_crf.masked_fill(labels_for_crf == -100, self.pad_label_id)
            crf_inputs = self.crf_dropout(emissions)
            crf_log_likelihood = self.crf(crf_inputs, labels_for_crf, mask)
            if crf_log_likelihood.dim() > 0:
                crf_log_likelihood = crf_log_likelihood.mean()
            loss = -crf_log_likelihood

        if return_predictions or labels is None:
            if mask is None:
                mask = torch.ones(emissions.size()[:2], dtype=torch.bool, device=emissions.device)
            crf_inputs = self.crf_dropout(emissions)
            batch_paths = self.crf.viterbi_decode(crf_inputs, mask)
            predictions = []
            for path in batch_paths:
                if hasattr(path, "tolist"):
                    path = path.tolist()
                predictions.append(path)

        return ModelOutput(
            loss=loss,
            logits=emissions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            crf_log_likelihood=crf_log_likelihood,
            predictions=predictions,
        )


def freeze_encoder_layers(model: DistilBertCrfForTokenClassification, num_layers_to_freeze: int) -> None:
    """Freeze the first ``num_layers_to_freeze`` Transformer layers for stability."""

    if num_layers_to_freeze <= 0:
        return

    encoder = model.distilbert.transformer
    for index in range(min(num_layers_to_freeze, len(encoder.layer))):
        for param in encoder.layer[index].parameters():
            param.requires_grad = False


def count_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable parameters, total parameters) for logging."""

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable_params, total_params
