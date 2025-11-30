# Configuration Notes

- `default.yaml`: Full dataset training (DistilBERT-CRF stabilization recipe: diff-LR/LLRD, EMA, R-Drop, gradual unfreeze).
- `sanity.yaml`: Lightweight variant for quick pipeline checks.
- `ablation/ema_off.yaml`: Turn off EMA (keep R-Drop on) for stabilization ablation.
- `ablation/rdrop_off.yaml`: Disable R-Drop (keep EMA on) for stabilization ablation.
- `ablation/aug_on.yaml`: Enable entity-aware augmentation with down-weighted loss.

## Planned Extensions

Add the following keys to experiment with upcoming features. Edit `default.yaml` or create new files (e.g., `char.yaml`, `gazetteer.yaml`) referencing these fields:

```yaml
model:
  use_char_features: false     # Enable char-CNN/BiLSTM embeddings
  char_hidden_size: 64
  use_gazetteer: false         # Fuse gazetteer logits via linear layer
  gazetteer_weight: 0.5

training:
  adversarial_training: null   # {"method": "fgm", "epsilon": 1.0}
  lora:
    enabled: false
    r: 8
    alpha: 32
    dropout: 0.1
```

When new fields are added, mirror the logic inside `modeling.py` / `trainer.py` and document the implementation in `docs/implementation_notes.md`.
