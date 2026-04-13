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

CONFIG="${CONFIG:-$ROOT_DIR/configs/text_conditional_eval/ETTh2_posterior.yaml}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
SAVE_GENERATED_DIR="${SAVE_GENERATED_DIR:-}"
CONDITIONAL_CKPT="${CONDITIONAL_CKPT:-}"
TS2VEC_DIR="${TS2VEC_DIR:-}"
USE_EMA_FOR_EVAL="${USE_EMA_FOR_EVAL:-1}"
SHUFFLE_CAPTION_EMBEDDINGS="${SHUFFLE_CAPTION_EMBEDDINGS:-0}"

CMD="python scripts/evaluate_text_conditional.py --config ${CONFIG} --split ${SPLIT}"

if [[ -n "${CONDITIONAL_CKPT}" ]]; then
  CMD="${CMD} --conditional-ckpt ${CONDITIONAL_CKPT}"
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

if [[ -n "${OUTPUT_JSON}" ]]; then
  CMD="${CMD} --output-json ${OUTPUT_JSON}"
fi

if [[ -n "${SAVE_GENERATED_DIR}" ]]; then
  CMD="${CMD} --save-generated-dir ${SAVE_GENERATED_DIR}"
fi

if [[ -n "${TS2VEC_DIR}" ]]; then
  CMD="${CMD} --ts2vec-dir ${TS2VEC_DIR}"
fi

if [[ "${USE_EMA_FOR_EVAL}" == "0" ]]; then
  CMD="${CMD} --no-ema-eval"
fi

if [[ "${SHUFFLE_CAPTION_EMBEDDINGS}" == "1" ]]; then
  CMD="${CMD} --shuffle-caption-embeddings"
fi

echo "[INFO] Running command:"
echo "${CMD}"
eval "${CMD}"
