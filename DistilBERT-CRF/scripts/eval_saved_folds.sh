#!/usr/bin/env bash
set -euo pipefail

# Evaluate existing fold checkpoints without re-training.
# Usage: ./scripts/eval_saved_folds.sh --config configs/ablation/aug_on.yaml --run-prefix ablate_aug_on [--checkpoint-tag best] [--folds 5] [--no-test]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
cd "${PROJECT_ROOT}"

CONFIG="configs/ablation/aug_on.yaml"
RUN_PREFIX="ablate_aug_on"
CHECKPOINT_TAG="best"
FOLDS=5
EVAL_TEST=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"; shift 2;;
    --run-prefix)
      RUN_PREFIX="$2"; shift 2;;
    --checkpoint-tag)
      CHECKPOINT_TAG="$2"; shift 2;;
    --folds)
      FOLDS="$2"; shift 2;;
    --no-test)
      EVAL_TEST=0; shift 1;;
    *)
      echo "Unknown argument: $1" >&2; exit 1;;
  esac
done

python_bin="${PYTHON:-python}"

for i in $(seq 1 "$FOLDS"); do
  run_name="${RUN_PREFIX}_fold${i}"
  metrics_out="training_logs/${run_name}_eval.json"

  cmd=("${python_bin}" "${PROJECT_ROOT}/scripts/train_distilbert_crf.py"
       --config "${CONFIG}" \
       --run-name "${run_name}" \
       --checkpoint-tag "${CHECKPOINT_TAG}" \
       --skip-training \
       --metrics-output "${metrics_out}")

  if [[ ${EVAL_TEST} -eq 1 ]]; then
    cmd+=(--evaluate-test)
  fi

  echo "Evaluating ${run_name} (tag=${CHECKPOINT_TAG})"
  "${cmd[@]}"
done

echo "Done. Metrics written to training_logs/${RUN_PREFIX}_fold*_eval.json"
