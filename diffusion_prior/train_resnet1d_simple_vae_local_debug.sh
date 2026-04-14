#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

echo "Running local single-GPU ResNet1D diffusion prior debug for SimpleVAE latents on host $(hostname)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

CONFIG="${CONFIG:-$ROOT_DIR/diffusion_prior/configs/resnet1d_simple_vae.yaml}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CONFIG=$CONFIG"

python "$ROOT_DIR/diffusion_prior/train_diffusion_prior.py" \
  --config "$CONFIG" \
  "$@"
