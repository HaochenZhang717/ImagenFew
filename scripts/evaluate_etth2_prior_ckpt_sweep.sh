#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-$ROOT_DIR/configs/self_cond_eval/ETTh2.yaml}"
PRIOR_DIR="${PRIOR_DIR:-$ROOT_DIR/logs/diffusion_prior/time_shift_sweep/ETTh2/tds_1p0}"
START_EPOCH="${START_EPOCH:-1000}"
END_EPOCH="${END_EPOCH:-8000}"
STEP_EPOCH="${STEP_EPOCH:-1000}"
SPLIT="${SPLIT:-train}"
SAMPLE_SOURCE="${SAMPLE_SOURCE:-prior}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/logs/self_conditional_generator/eval/ETTh2/prior_ckpt_sweep}"
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
echo "[INFO] PRIOR_DIR=$PRIOR_DIR"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] Evaluating checkpoints from $START_EPOCH to $END_EPOCH every $STEP_EPOCH epochs"

for (( epoch=START_EPOCH; epoch<=END_EPOCH; epoch+=STEP_EPOCH )); do
  PRIOR_CKPT="$PRIOR_DIR/diffusion_prior_epoch_$(printf "%04d" "$epoch").pt"
  if [[ ! -f "$PRIOR_CKPT" ]]; then
    echo "[WARN] Missing checkpoint, skipping: $PRIOR_CKPT"
    continue
  fi

  OUTPUT_JSON="$OUTPUT_DIR/epoch_$(printf "%04d" "$epoch").json"
  SAVE_GENERATED_DIR=""
  if [[ -n "$SAVE_GENERATED_ROOT" ]]; then
    SAVE_GENERATED_DIR="$SAVE_GENERATED_ROOT/epoch_$(printf "%04d" "$epoch")"
    mkdir -p "$SAVE_GENERATED_DIR"
  fi

  CMD=(
    python
    scripts/evaluate_self_conditional.py
    --config "$CONFIG"
    --split "$SPLIT"
    --sample-source "$SAMPLE_SOURCE"
    --prior-ckpt "$PRIOR_CKPT"
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

  printf '[INFO] Evaluating epoch %s with command:\n%s\n' "$epoch" "${CMD[*]}"
  "${CMD[@]}"

  python - "$OUTPUT_JSON" "$epoch" >> "$SUMMARY_JSONL" <<'PY'
import json
import sys

path = sys.argv[1]
epoch = int(sys.argv[2])
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
record = {"epoch": epoch, **obj}
print(json.dumps(record))
PY
done

echo "[INFO] Sweep finished."
echo "[INFO] Per-checkpoint json files: $OUTPUT_DIR"
echo "[INFO] Summary jsonl: $SUMMARY_JSONL"
