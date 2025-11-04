# Configuration Notes

- `default.yaml`: Full dataset training (batch_size=16, num_epochs=10, DistilBERT-CRF baseline).
- `sanity.yaml`: Lightweight variant for quick pipeline checks.

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
