#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CAPTION_DIR="${CAPTION_DIR:-/playpen-shared/haochenz/ImagenFew/logs/finetune_captions/ETTh2}"
SAVE_PATH="${SAVE_PATH:-/playpen-shared/haochenz/ImagenFew/logs/finetune_captions/ETTh2/train_caption_embeds.pt}"
SPLIT="${SPLIT:-train}"
NUM_PARTS="${NUM_PARTS:-4}"
N_VARS="${N_VARS:-7}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-Embedding-2B}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"

python scripts/precompute_embeds.py \
  --caption-dir "$CAPTION_DIR" \
  --split "$SPLIT" \
  --num-parts "$NUM_PARTS" \
  --n-vars "$N_VARS" \
  --save-path "$SAVE_PATH" \
  --batch-size "$BATCH_SIZE" \
  --model-name "$MODEL_NAME" \
  --torch-dtype "$TORCH_DTYPE"
