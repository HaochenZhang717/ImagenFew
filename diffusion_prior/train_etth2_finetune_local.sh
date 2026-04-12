#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

echo "Running local 4-GPU diffusion prior finetune for ETTh2 on host $(hostname)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29671}"
CONFIG="${CONFIG:-$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_etth2.yaml}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MASTER_PORT=$MASTER_PORT"
echo "CONFIG=$CONFIG"

torchrun --standalone --master_port="$MASTER_PORT" --nproc_per_node="$NPROC_PER_NODE" \
  "$ROOT_DIR/diffusion_prior/train_diffusion_prior.py" \
  --config "$CONFIG" \
  "$@"
