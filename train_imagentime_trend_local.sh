#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-./configs/ImagenTimeDecomposed/ETTh2_trend.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-ImagenTime-Decomposed}"
SUBSET_P="${SUBSET_P:-1.0}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python run.py \
  --subset_p "$SUBSET_P" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --config "$CONFIG" \
  "$@"
