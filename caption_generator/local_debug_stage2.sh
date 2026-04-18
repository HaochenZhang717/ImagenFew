#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/ettm1_stage2_diffusion_prior.yaml}"
DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"

HF_HOME_DEFAULT="$HOME/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ "${DEVICE}" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./logs/caption_generator/ettm1_stage2_diffusion_prior_local_debug}"
STAGE1_CONFIG_PATH="${STAGE1_CONFIG_PATH:-./configs/ettm1_stage1_qwen25_3b.yaml}"
STAGE1_CKPT_PATH="${STAGE1_CKPT_PATH:-./logs/caption_generator/ettm1_stage1_qwen25_3b/joint_caption_best.pt}"

NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAMPLE_EVERY_EPOCHS="${SAMPLE_EVERY_EPOCHS:-1}"
NUM_DECODE_SAMPLES="${NUM_DECODE_SAMPLES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
WANDB="${WANDB:-0}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ ! -f "${STAGE1_CONFIG_PATH}" ]]; then
  echo "[ERROR] Stage 1 config not found: ${STAGE1_CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${STAGE1_CKPT_PATH}" ]]; then
  echo "[ERROR] Stage 1 checkpoint not found: ${STAGE1_CKPT_PATH}" >&2
  echo "[HINT] Override it with STAGE1_CKPT_PATH=/path/to/joint_caption_best.pt" >&2
  exit 1
fi

OVERRIDES=(
  "seed=${SEED}"
  "device=${DEVICE}"
  "output_dir=${OUTPUT_DIR}"
  "stage1.config_path=${STAGE1_CONFIG_PATH}"
  "stage1.checkpoint_path=${STAGE1_CKPT_PATH}"
  "training.num_epochs=${NUM_EPOCHS}"
  "training.batch_size=${BATCH_SIZE}"
  "training.eval_batch_size=${EVAL_BATCH_SIZE}"
  "data.encode_batch_size=${ENCODE_BATCH_SIZE}"
  "data.num_workers=${NUM_WORKERS}"
  "sampling.sample_every_epochs=${SAMPLE_EVERY_EPOCHS}"
  "sampling.num_decode_samples=${NUM_DECODE_SAMPLES}"
  "sampling.max_new_tokens=${MAX_NEW_TOKENS}"
)

if [[ "${WANDB}" == "1" ]]; then
  OVERRIDES+=("wandb.enabled=true")
  OVERRIDES+=("wandb.mode=online")
else
  OVERRIDES+=("wandb.enabled=false")
fi

if [[ "$#" -gt 0 ]]; then
  OVERRIDES+=("$@")
fi

CMD=(
  python
  train_stage2.py
  --config "${CONFIG}"
  --override
)

CMD+=("${OVERRIDES[@]}")

echo "[INFO] Running Stage 2 local debug"
echo "[INFO] HF_HOME=${HF_HOME}"
if [[ "${DEVICE}" == "cuda" ]]; then
  echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi
printf '[INFO] Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
