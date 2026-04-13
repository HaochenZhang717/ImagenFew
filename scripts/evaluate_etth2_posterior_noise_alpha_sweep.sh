#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/self_cond_eval/ETTh2.yaml}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"
ALPHAS="${ALPHAS:-0.0 0.05 0.1 0.2 0.3 0.5 0.7 1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/self_conditional_generator/eval/ETTh2/posterior_noise_alpha_sweep}"
SAVE_GENERATED_ROOT="${SAVE_GENERATED_ROOT:-}"
CONDITIONAL_CKPT="${CONDITIONAL_CKPT:-}"

mkdir -p "$OUTPUT_DIR"
SUMMARY_JSONL="$OUTPUT_DIR/summary.jsonl"
: > "$SUMMARY_JSONL"

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

echo "[INFO] CONFIG=$CONFIG"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] ALPHAS=$ALPHAS"
echo "[INFO] SPLIT=$SPLIT"

for alpha in $ALPHAS; do
  alpha_tag="${alpha//./p}"
  OUTPUT_JSON="$OUTPUT_DIR/${SPLIT}_alpha_${alpha_tag}.json"
  SAVE_GENERATED_DIR=""
  if [[ -n "$SAVE_GENERATED_ROOT" ]]; then
    SAVE_GENERATED_DIR="$SAVE_GENERATED_ROOT/alpha_${alpha_tag}"
    mkdir -p "$SAVE_GENERATED_DIR"
  fi

  CMD=(
    python
    scripts/evaluate_self_conditional.py
    --config "$CONFIG"
    --split "$SPLIT"
    --sample-source posterior
    --posterior-noise-alpha "$alpha"
    --output-json "$OUTPUT_JSON"
  )

  if [[ -n "$CONDITIONAL_CKPT" ]]; then
    CMD+=(--conditional-ckpt "$CONDITIONAL_CKPT")
  fi

  if [[ -n "$MAX_SAMPLES" ]]; then
    CMD+=(--max-samples "$MAX_SAMPLES")
  fi

  if [[ -n "$SAVE_GENERATED_DIR" ]]; then
    CMD+=(--save-generated-dir "$SAVE_GENERATED_DIR")
  fi

  if [[ "$USE_EMA_FOR_EVAL" == "0" ]]; then
    CMD+=(--no-ema-eval)
  fi

  printf '[INFO] Evaluating alpha %s with command:\n%s\n' "$alpha" "${CMD[*]}"
  "${CMD[@]}"

  python - "$OUTPUT_JSON" "$alpha" >> "$SUMMARY_JSONL" <<'PY'
import json
import sys

path = sys.argv[1]
alpha = float(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
record = {"alpha": alpha, **obj}
print(json.dumps(record))
PY
done

echo "[INFO] Alpha sweep finished."
echo "[INFO] Per-alpha json files: $OUTPUT_DIR"
echo "[INFO] Summary jsonl: $SUMMARY_JSONL"
