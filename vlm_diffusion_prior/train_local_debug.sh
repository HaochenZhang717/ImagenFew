#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-ETTh2}"
GPU_ID="${GPU_ID:-0}"
PRECISION="${PRECISION:-bf16}"
WANDB="${WANDB:-0}"
COMPILE="${COMPILE:-0}"
CKPT="${CKPT:-}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"
NUM_GPUS=1

case "${DATASET}" in
  ETTh2)
    CONFIG="${CONFIG:-DiTDH-S-ETTh2.yaml}"
    IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/../logs/finetune_dataset_images/ETTh2/train}"
    RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/../logs/vlm_diffusion_prior/ETTh2/DiTDH-S-debug}"
    PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-$ROOT_DIR/../logs/vlm_diffusion_prior/ETTh2/precomputed_vision_embeds}"
    NUM_CH="${NUM_CH:-7}"
    MASTER_PORT="${MASTER_PORT:-29678}"
    ;;
  mujoco)
    CONFIG="${CONFIG:-DiTDH-S-mujoco.yaml}"
    IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/../logs/finetune_dataset_images/mujoco/train}"
    RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/../logs/vlm_diffusion_prior/mujoco/DiTDH-S-debug}"
    PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-$ROOT_DIR/../logs/vlm_diffusion_prior/mujoco/precomputed_vision_embeds}"
    NUM_CH="${NUM_CH:-14}"
    MASTER_PORT="${MASTER_PORT:-29679}"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}. Use ETTh2 or mujoco." >&2
    exit 1
    ;;
esac

USE_PRECOMPUTED="${USE_PRECOMPUTED:-0}"

export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"
export TOKENIZERS_PARALLELISM=false
HOUSTON_TIMESTAMP="$(TZ=America/Chicago date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$HOUSTON_TIMESTAMP}"
export ENTITY="${ENTITY:-zhanghaochen}"
export PROJECT="${PROJECT:-VLM-Diffusion-Prior}"

CMD="torchrun --master_port=${MASTER_PORT} --nproc_per_node=${NUM_GPUS} train.py \
  --config ${CONFIG} \
  --image-path ${IMAGE_PATH} \
  --num-ch ${NUM_CH} \
  --results-dir ${RESULTS_DIR} \
  --precision ${PRECISION} \
  --global-seed ${GLOBAL_SEED}"

if [[ "${USE_PRECOMPUTED}" == "1" ]]; then
  CMD="${CMD} --precomputed-dir ${PRECOMPUTED_DIR}"
fi

if [[ "${WANDB}" == "1" ]]; then
  CMD="${CMD} --wandb"
fi

if [[ "${COMPILE}" == "1" ]]; then
  CMD="${CMD} --compile"
fi

if [[ -n "${CKPT}" ]]; then
  CMD="${CMD} --ckpt ${CKPT}"
fi

echo "[INFO] DATASET=${DATASET}"
echo "[INFO] Running command:"
echo "${CMD}"
eval "${CMD}"
