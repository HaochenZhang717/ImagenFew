#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/ImagenFew"
SCRIPT="$ROOT_DIR/scripts_imagen_few/sample_imagenfew_local.sh"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"
EVAL_METRICS="${EVAL_METRICS:-disc contextFID}"
TS2VEC_DIR="${TS2VEC_DIR:-}"
SEED="${SEED:-}"

CONDA_ENV="vlm" \
TREND_ONLY="1" \
MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/ETTh2_trend_only/3b0d5aa8-14a2-46f9-9ff2-eced98ed6d02/ImagenFew.pt" \
CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/ETTh2.yaml" \
DATASET="ETTh2" \
SPLIT="$SPLIT" \
MAX_SAMPLES="$MAX_SAMPLES" \
EVAL_METRICS="$EVAL_METRICS" \
TS2VEC_DIR="$TS2VEC_DIR" \
SEED="$SEED" \
USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
bash "$SCRIPT"

CONDA_ENV="vlm" \
TREND_ONLY="1" \
MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/AirQuality_trend_only/e912f5d3-4654-485f-aebe-b674c1225210/ImagenFew.pt" \
CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/AirQuality.yaml" \
DATASET="AirQuality" \
SPLIT="$SPLIT" \
MAX_SAMPLES="$MAX_SAMPLES" \
EVAL_METRICS="$EVAL_METRICS" \
TS2VEC_DIR="$TS2VEC_DIR" \
SEED="$SEED" \
USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
bash "$SCRIPT"

CONDA_ENV="vlm" \
TREND_ONLY="1" \
MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/Mujoco_trend_only/566bd493-c193-43ce-bf32-9cecad62054c/ImagenFew.pt" \
CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/Mujoco.yaml" \
DATASET="mujoco" \
SPLIT="$SPLIT" \
MAX_SAMPLES="$MAX_SAMPLES" \
EVAL_METRICS="$EVAL_METRICS" \
TS2VEC_DIR="$TS2VEC_DIR" \
SEED="$SEED" \
USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
bash "$SCRIPT"
