#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/finetune/ETTh2.yaml}"
MODEL_CKPT="${MODEL_CKPT:-}"
DATASET="${DATASET:-}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT="${OUTPUT:-}"
SEED="${SEED:-}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"

if [[ -z "$MODEL_CKPT" ]]; then
  echo "ERROR: Please provide MODEL_CKPT=/path/to/ImagenFew.pt" >&2
  exit 1
fi

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
    eval "$($CONDA_BIN shell.bash hook)"
    conda activate "$CONDA_ENV"
  fi
fi

CMD=(
  python
  "$SCRIPT_DIR/sample_imagenfew.py"
  --config "$CONFIG"
  --model-ckpt "$MODEL_CKPT"
  --split "$SPLIT"
)

if [[ -n "$DATASET" ]]; then
  CMD+=(--dataset "$DATASET")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ -n "$OUTPUT" ]]; then
  CMD+=(--output "$OUTPUT")
fi

if [[ -n "$SEED" ]]; then
  CMD+=(--seed "$SEED")
fi

if [[ "$USE_EMA_FOR_EVAL" == "0" ]]; then
  CMD+=(--no-ema-eval)
fi

echo "[INFO] Running command:"
echo "${CMD[*]}"
"${CMD[@]}"
