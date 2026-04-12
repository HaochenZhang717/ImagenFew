#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/self_cond_eval/ETTh2.yaml}"
SPLIT="${SPLIT:-train}"
SAMPLE_SOURCE="${SAMPLE_SOURCE:-both}"
DATASET="${DATASET:-}"
CONDITIONAL_CKPT="${CONDITIONAL_CKPT:-}"
PRIOR_CKPT="${PRIOR_CKPT:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SAVE_GENERATED_DIR="${SAVE_GENERATED_DIR:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

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

CMD=(
  python
  scripts/evaluate_self_conditional.py
  --config "$CONFIG"
  --split "$SPLIT"
  --sample-source "$SAMPLE_SOURCE"
)

if [[ -n "$DATASET" ]]; then
  CMD+=(--dataset "$DATASET")
fi

if [[ -n "$CONDITIONAL_CKPT" ]]; then
  CMD+=(--conditional-ckpt "$CONDITIONAL_CKPT")
fi

if [[ -n "$PRIOR_CKPT" ]]; then
  CMD+=(--prior-ckpt "$PRIOR_CKPT")
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

printf '[INFO] Running command:\n%s\n' "${CMD[*]}"
"${CMD[@]}"
