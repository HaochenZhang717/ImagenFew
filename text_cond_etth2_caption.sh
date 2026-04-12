#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/text_conditional_generator/ETTh2_caption.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-TextConditionalGeneration}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BIN=""
  if [[ -x "/playpen/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen/haochenz/miniconda3/bin/conda"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  fi

  if [[ -n "$CONDA_BIN" ]]; then
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$CONDA_ENV"
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"

CMD=(
  python
  run.py
  --subset_p 1.0
  --wandb
  --wandb_project "$WANDB_PROJECT"
  --config "$CONFIG"
)

if [[ "$NUM_GPUS" -gt 1 ]]; then
  CMD=(
    torchrun
    --standalone
    --nproc_per_node="$NUM_GPUS"
    run.py
    --ddp
    --subset_p 1.0
    --wandb
    --wandb_project "$WANDB_PROJECT"
    --config "$CONFIG"
  )
fi

printf '[INFO] Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
