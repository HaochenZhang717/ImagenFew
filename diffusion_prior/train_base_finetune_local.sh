#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

echo "Running local diffusion prior base finetune on host $(hostname)"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"


#CONFIGS=(
#  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_airquality.yaml"
#  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_etth2.yaml"
#  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_mujoco.yaml"
#)

CONFIGS=(
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_weather.yaml"
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_starlightcurves.yaml"
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_sine.yaml"
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_selfregulationscp1.yaml"
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_saugeenriverflow.yaml"
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_ili.yaml"
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_ettm1.yaml"
  "$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_ettm2.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Starting finetune with config: $config"
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$ROOT_DIR/diffusion_prior/train_diffusion_prior.py" \
    --config "$config" \
    "$@"
done
