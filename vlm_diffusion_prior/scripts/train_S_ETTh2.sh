#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NUM_GPUS="${NUM_GPUS:-2}"
CONFIG="${CONFIG:-DiTDH-S-ETTh2.yaml}"
IMAGE_PATH="${IMAGE_PATH:-$ROOT_DIR/../logs/finetune_dataset_images/ETTh2/train}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/../logs/vlm_diffusion_prior/ETTh2/DiTDH-S}"
NUM_CH="${NUM_CH:-7}"

export HF_HOME="${HF_HOME:-/playpen/haochenz/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DiTDH-S-ETTh2}"
export ENTITY="${ENTITY:-zhanghaochen}"
export PROJECT="${PROJECT:-VLM-Diffusion-Prior}"

PRECISION="${PRECISION:-bf16}"
WANDB="${WANDB:-1}"
COMPILE="${COMPILE:-1}"
CKPT="${CKPT:-}"
GLOBAL_SEED="${GLOBAL_SEED:-42}"

CMD="torchrun --master_port=${MASTER_PORT:-29658} --nproc_per_node=${NUM_GPUS} train.py \
  --config ${CONFIG} \
  --image-path ${IMAGE_PATH} \
  --num-ch ${NUM_CH} \
  --results-dir ${RESULTS_DIR} \
  --precision ${PRECISION} \
  --global-seed ${GLOBAL_SEED}"

if [[ "${WANDB}" == "1" ]]; then
  CMD="${CMD} --wandb"
fi

if [[ "${COMPILE}" == "1" ]]; then
  CMD="${CMD} --compile"
fi

if [[ -n "${CKPT}" ]]; then
  CMD="${CMD} --ckpt ${CKPT}"
fi

echo "[INFO] Running command:"
echo "${CMD}"
eval "${CMD}"
