#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/conditional_imagen_few_trend/ETTh2_seasonal.yaml}"
SPLIT="${SPLIT:-train}"
SAMPLE_SOURCE="${SAMPLE_SOURCE:-prior}"
MODEL_CKPT="${MODEL_CKPT:-}"
TREND_MODEL_CKPT="${TREND_MODEL_CKPT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAVE_GENERATED_DIR="${SAVE_GENERATED_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"

CMD=(
  python
  scripts/evaluate_imagenfew_cross_attention.py
  --config "$CONFIG"
  --split "$SPLIT"
  --sample-source "$SAMPLE_SOURCE"
)

if [[ -n "$MODEL_CKPT" ]]; then
  CMD+=(--model-ckpt "$MODEL_CKPT")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$SAVE_GENERATED_DIR" ]]; then
  CMD+=(--save-generated-dir "$SAVE_GENERATED_DIR")
fi
if [[ -n "$OUTPUT_JSON" ]]; then
  CMD+=(--output-json "$OUTPUT_JSON")
fi
if [[ "$USE_EMA_FOR_EVAL" == "0" ]]; then
  CMD+=(--no-ema-eval)
fi

if [[ -n "$TREND_MODEL_CKPT" ]]; then
  export TREND_MODEL_CKPT
fi

printf '[INFO] Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
