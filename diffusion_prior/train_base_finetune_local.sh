#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

echo "Running local diffusion prior base finetune on host $(hostname)"
GPU_IDS=(${GPU_IDS:-0 1 2 3})
LOG_DIR="$ROOT_DIR/logs/diffusion_prior_finetune"
mkdir -p "$LOG_DIR"

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

for start in $(seq 0 ${#GPU_IDS[@]} $((${#CONFIGS[@]} - 1))); do
  pids=()
  for gpu_idx in "${!GPU_IDS[@]}"; do
    config_idx=$((start + gpu_idx))
    if (( config_idx >= ${#CONFIGS[@]} )); then
      break
    fi

    gpu="${GPU_IDS[$gpu_idx]}"
    config="${CONFIGS[$config_idx]}"
    name="$(basename "$config" .yaml)"
    log_file="$LOG_DIR/${name}_gpu${gpu}.log"

    echo "Starting $name on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT_DIR/diffusion_prior/train_diffusion_prior.py" \
      --config "$config" \
      "$@" >"$log_file" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
