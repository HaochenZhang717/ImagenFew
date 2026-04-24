#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  else
    echo "CONDA_ENV is set, but 'conda' is not available in PATH." >&2
    exit 1
  fi
fi

CONFIG="${CONFIG:-./configs/ImagenTimeVectorCond/VerbalTS_istanbul_traffic_dit_minilm.yaml}"
GPU_ID="${GPU_ID:-0}"
SUBSET_P="${SUBSET_P:-1.0}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-ImagenTimeVectorCond-VerbalTS}"
RUN_SUFFIX="${RUN_SUFFIX:-local_dit}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-}"
LABEL_DROPOUT="${LABEL_DROPOUT:-}"
LR_OVERRIDE="${LR_OVERRIDE:-}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

config_base="$(basename "$CONFIG" .yaml)"
run_id="${config_base}_${RUN_SUFFIX}_$(date +%m%d_%H%M%S)"

CMD=(
  python run_diffts_imagentime.py
  --config "$CONFIG"
  --run_id "$run_id"
  --subset_p "$SUBSET_P"
)

if [[ -n "$GUIDANCE_SCALE" ]]; then
  CMD+=(--guidance_scale "$GUIDANCE_SCALE")
fi

if [[ -n "$LABEL_DROPOUT" ]]; then
  CMD+=(--label_dropout "$LABEL_DROPOUT")
fi

if [[ -n "$LR_OVERRIDE" ]]; then
  CMD+=(--learning_rate "$LR_OVERRIDE")
fi

if [[ -n "$BATCH_SIZE_OVERRIDE" ]]; then
  CMD+=(--batch_size "$BATCH_SIZE_OVERRIDE")
fi

if [[ -n "$EPOCHS_OVERRIDE" ]]; then
  CMD+=(--epochs "$EPOCHS_OVERRIDE")
fi

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"
echo "Run ID: $run_id"
printf 'Running command:\n  %q' "${CMD[0]}"
for arg in "${CMD[@]:1}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${CMD[@]}"
