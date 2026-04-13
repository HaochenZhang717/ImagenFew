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

CONFIG="${CONFIG:-$ROOT_DIR/configs/self_cond_eval/ETTh2.yaml}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

CMD="python scripts/compare_conditional_latent_stats.py --config ${CONFIG} --split ${SPLIT} --batch-size ${BATCH_SIZE}"

if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD="${CMD} --output-json ${OUTPUT_JSON}"
fi

echo "[INFO] Running command:"
echo "${CMD}"
eval "${CMD}"
