#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/self_cond_eval/ETTh2.yaml}"
PRIOR_CKPT="${PRIOR_CKPT:-$ROOT_DIR/logs/diffusion_prior/time_shift_sweep/ETTh2/tds_1p0/diffusion_prior_latest.pt}"
CONDITIONAL_CKPT="${CONDITIONAL_CKPT:-}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
PRIOR_NUM_STEPS="${PRIOR_NUM_STEPS:-50}"
PRIOR_ATOL="${PRIOR_ATOL:-1e-6}"
PRIOR_RTOL="${PRIOR_RTOL:-1e-3}"
METHODS="${METHODS:-euler midpoint rk4 adaptive_heun bosh3 dopri5 dopri8}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/self_conditional_generator/eval/ETTh2/prior_sampler_sweep}"
SAVE_GENERATED_ROOT="${SAVE_GENERATED_ROOT:-}"

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
echo "[INFO] PRIOR_CKPT=$PRIOR_CKPT"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] METHODS=$METHODS"
echo "[INFO] Using EMA model for eval and latest prior checkpoint."

for method in $METHODS; do
  OUTPUT_JSON="$OUTPUT_DIR/${SPLIT}_${method}.json"
  SAVE_GENERATED_DIR=""
  if [[ -n "$SAVE_GENERATED_ROOT" ]]; then
    SAVE_GENERATED_DIR="$SAVE_GENERATED_ROOT/$method"
    mkdir -p "$SAVE_GENERATED_DIR"
  fi

  CMD=(
    python
    scripts/evaluate_self_conditional.py
    --config "$CONFIG"
    --split "$SPLIT"
    --sample-source prior
    --prior-ckpt "$PRIOR_CKPT"
    --prior-sampling-method "$method"
    --prior-num-steps "$PRIOR_NUM_STEPS"
    --prior-atol "$PRIOR_ATOL"
    --prior-rtol "$PRIOR_RTOL"
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

  printf '[INFO] Evaluating method %s with command:\n%s\n' "$method" "${CMD[*]}"
  "${CMD[@]}"

  python - "$OUTPUT_JSON" "$method" >> "$SUMMARY_JSONL" <<'PY'
import json
import sys

path = sys.argv[1]
method = sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
record = {"method": method, **obj}
print(json.dumps(record))
PY
done

echo "[INFO] Sampler sweep finished."
echo "[INFO] Per-method json files: $OUTPUT_DIR"
echo "[INFO] Summary jsonl: $SUMMARY_JSONL"
