#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BIN=""
  if [[ -x "/playpen/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen/haochenz/miniconda3/bin/conda"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  fi

  if [[ -n "$CONDA_BIN" ]]; then
    eval "$("$CONDA_BIN" shell.bash hook)"
    conda activate "$CONDA_ENV"
  fi
fi

CONFIG="${CONFIG:-$ROOT_DIR/configs/conditional_imagen_few_eval/ETTh2_prior.yaml}"
SPLIT="${SPLIT:-train}"
SAMPLE_SOURCE="${SAMPLE_SOURCE:-prior}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
SAVE_GENERATED_DIR="${SAVE_GENERATED_DIR:-}"
MODEL_CKPT="${MODEL_CKPT:-}"
PRIOR_CKPT="${PRIOR_CKPT:-}"
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

if [[ -n "$PRIOR_CKPT" ]]; then
  CMD+=(--prior-ckpt "$PRIOR_CKPT")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$OUTPUT_JSON" ]]; then
  CMD+=(--output-json "$OUTPUT_JSON")
fi

if [[ -n "$SAVE_GENERATED_DIR" ]]; then
  CMD+=(--save-generated-dir "$SAVE_GENERATED_DIR")
fi

if [[ "$USE_EMA_FOR_EVAL" == "0" ]]; then
  CMD+=(--no-ema-eval)
fi

echo "[INFO] Running command:"
echo "${CMD[*]}"
"${CMD[@]}"
