#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/ImagenFew"
SCRIPT="$ROOT_DIR/scripts_imagen_few/sample_imagenfew_corrupted.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SEED="${SEED:-}"
NUM_CORRUPTIONS="${NUM_CORRUPTIONS:-5}"
MIN_STEP_FRAC="${MIN_STEP_FRAC:-0.05}"
MAX_STEP_FRAC="${MAX_STEP_FRAC:-0.95}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"

COMMON_ARGS=()
if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$SEED" ]]; then
  COMMON_ARGS+=(--seed "$SEED")
fi
if [[ "$USE_EMA_FOR_EVAL" != "1" ]]; then
  COMMON_ARGS+=(--no-ema-eval)
fi

run_one() {
  local config="$1"
  local ckpt="$2"
  local dataset="$3"

  "$PYTHON_BIN" "$SCRIPT" \
    --config "$config" \
    --model-ckpt "$ckpt" \
    --dataset "$dataset" \
    --split "$SPLIT" \
    --trend_only \
    --num-corruptions "$NUM_CORRUPTIONS" \
    --min-step-frac "$MIN_STEP_FRAC" \
    --max-step-frac "$MAX_STEP_FRAC" \
    "${COMMON_ARGS[@]}"
}

run_one \
  "/playpen-shared/haochenz/ImagenFew/configs/finetune/ETTh2.yaml" \
  "/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/ETTh2_trend_only/3b0d5aa8-14a2-46f9-9ff2-eced98ed6d02/ImagenFew.pt" \
  "ETTh2"

run_one \
  "/playpen-shared/haochenz/ImagenFew/configs/finetune/AirQuality.yaml" \
  "/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/AirQuality_trend_only/e912f5d3-4654-485f-aebe-b674c1225210/ImagenFew.pt" \
  "AirQuality"


run_one \
  "/playpen-shared/haochenz/ImagenFew/configs/finetune/Mujoco.yaml" \
  "/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/Mujoco_trend_only/566bd493-c193-43ce-bf32-9cecad62054c/ImagenFew.pt" \
  "mujoco"
