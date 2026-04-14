#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

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

CONFIG="${CONFIG:-./configs/finetune_simple_vae/ETTh2.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-SimpleVAE-Finetune}"
SUBSET_P="${SUBSET_P:-1.0}"

echo "Running SimpleVAE finetune-from-scratch locally on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "CONFIG=$CONFIG"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "SUBSET_P=$SUBSET_P"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python run_no_sample.py \
  --subset_p "$SUBSET_P" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --config "$CONFIG" \
  "$@"
