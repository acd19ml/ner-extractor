#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
source "${PROJECT_ROOT}/../.venv/bin/activate"
python "${PROJECT_ROOT}/scripts/train_distilbert_crf.py" \
  --config "${PROJECT_ROOT}/configs/default.yaml" \
  --run-name distilbert_crf_baseline \
  --skip-training \
  --evaluate-test > "${PROJECT_ROOT}/training_logs/distilbert_crf_baseline_eval.txt" 2>&1
