#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/playpen-shared/haochenz/ImagenFew"
SCRIPT="$ROOT_DIR/scripts_imagen_few/sample_imagenfew_local.sh"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"
EVAL_METRICS="${EVAL_METRICS:-disc contextFID}"


# sample time series

CONDA_ENV="vlm" \
MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/ETTh2/404bb2bf-e4b2-412c-87b5-032bac01e4f9/ImagenFew.pt" \
CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/ETTh2.yaml" \
SPLIT="train" \
MAX_SAMPLES="$MAX_SAMPLES" \
EVAL_METRICS="$EVAL_METRICS" \
USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
bash /playpen-shared/haochenz/ImagenFew/scripts_imagen_few/sample_imagenfew_local.sh

CONDA_ENV="vlm" \
MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/AirQuality/d040753f-6f2d-4c58-bae3-ff2764e47492/ImagenFew.pt" \
CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/AirQuality.yaml" \
SPLIT="train" \
MAX_SAMPLES="$MAX_SAMPLES" \
EVAL_METRICS="$EVAL_METRICS" \
USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
bash /playpen-shared/haochenz/ImagenFew/scripts_imagen_few/sample_imagenfew_local.sh


CONDA_ENV="vlm" \
MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/Mujoco/faed7452-4f59-41dd-9bb9-882467dcb5b0/ImagenFew.pt" \
CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/Mujoco.yaml" \
SPLIT="train" \
MAX_SAMPLES="$MAX_SAMPLES" \
EVAL_METRICS="$EVAL_METRICS" \
USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
bash /playpen-shared/haochenz/ImagenFew/scripts_imagen_few/sample_imagenfew_local.sh



# sample trend only time series
#CONDA_ENV="vlm" \
#MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/ETTh2_trend_only/3b0d5aa8-14a2-46f9-9ff2-eced98ed6d02/ImagenFew.pt" \
#CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/ETTh2.yaml" \
#SPLIT="train" \
#MAX_SAMPLES="$MAX_SAMPLES" \
#EVAL_METRICS="$EVAL_METRICS" \
#USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
#bash /playpen-shared/haochenz/ImagenFew/scripts_imagen_few/sample_imagenfew_local.sh
#
#CONDA_ENV="vlm" \
#MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/AirQuality/d040753f-6f2d-4c58-bae3-ff2764e47492/ImagenFew.pt" \
#CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/AirQuality.yaml" \
#SPLIT="train" \
#MAX_SAMPLES="$MAX_SAMPLES" \
#EVAL_METRICS="$EVAL_METRICS" \
#USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
#bash /playpen-shared/haochenz/ImagenFew/scripts_imagen_few/sample_imagenfew_local.sh
#
#
#CONDA_ENV="vlm" \
#MODEL_CKPT="/playpen-shared/haochenz/ImagenFew/logs/ImagenFew/Mujoco/faed7452-4f59-41dd-9bb9-882467dcb5b0/ImagenFew.pt" \
#CONFIG="/playpen-shared/haochenz/ImagenFew/configs/finetune/Mujoco.yaml" \
#SPLIT="train" \
#MAX_SAMPLES="$MAX_SAMPLES" \
#EVAL_METRICS="$EVAL_METRICS" \
#USE_EMA_FOR_EVAL="$USE_EMA_FOR_EVAL" \
#bash /playpen-shared/haochenz/ImagenFew/scripts_imagen_few/sample_imagenfew_local.sh


