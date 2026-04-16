#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/ImagenFew"
SCRIPT="$ROOT_DIR/scripts_imagen_few/sample_imagenfew_local.sh"
MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
CONDA_ENV="${CONDA_ENV:-vlm}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"
EVAL_METRICS="${EVAL_METRICS:-disc contextFID pred}"


for dataset in AirQuality ETTh2 Mujoco; do
  CONFIG="$ROOT_DIR/configs/finetune/${dataset}.yaml"
  echo "[INFO] Sampling $dataset"
  CONDA_ENV="$CONDA_ENV"   MODEL_CKPT="$MODEL_CKPT"   CONFIG="$CONFIG"   SPLIT="$SPLIT"   MAX_SAMPLES="$MAX_SAMPLES"   EVAL_METRICS="$EVAL_METRICS"   USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL"   bash "$SCRIPT"
done
