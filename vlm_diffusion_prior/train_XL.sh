#!/bin/bash
set -e

# ==========================
# User Config
# ==========================
NUM_GPUS=8
CONFIG="DiTDH-XL.yaml"

IMAGE_PATH="/playpen-shared/haochenz/image_synthetic_u_square"
JSONL_PATH="../step_1_dataset_construction/synthetic_u_caption/time_series_caps_3072.jsonl"
RESULTS_DIR="/playpen-shared/haochenz/ckpts_diffusion_prior"


PRECISION="bf16"   # fp32 / fp16 / bf16
WANDB=0            # 1 enable, 0 disable
COMPILE=1          # 1 enable, 0 disable
CKPT=""            # optional resume checkpoint
GLOBAL_SEED=0

# ==========================
# Env
# ==========================
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8

# reduce tokenizer parallel warning
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_NAME="DiTDH-XL"
export ENTITY="zhanghaochen"
export PROJECT="Step2-Diffusion-Prior"
# optional debug flags
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# ==========================
# Build command
# ==========================
MASTER_PORT=29501

CMD="torchrun --master_port=${MASTER_PORT} --nproc_per_node=${NUM_GPUS} train.py \
  --config ${CONFIG} \
  --image-path ${IMAGE_PATH} \
  --jsonl-path ${JSONL_PATH} \
  --results-dir ${RESULTS_DIR} \
  --precision ${PRECISION} \
  --global-seed ${GLOBAL_SEED}"

if [ ${WANDB} -eq 1 ]; then
  CMD="${CMD} --wandb"
fi

if [ ${COMPILE} -eq 1 ]; then
  CMD="${CMD} --compile"
fi

if [ ! -z "${CKPT}" ]; then
  CMD="${CMD} --ckpt ${CKPT}"
fi

# ==========================
# Run
# ==========================
echo "[INFO] Running command:"
echo "${CMD}"
eval "${CMD}"