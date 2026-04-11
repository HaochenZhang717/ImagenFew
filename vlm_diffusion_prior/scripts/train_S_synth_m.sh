#!/bin/bash
set -e

# ==========================
# User Config
# ==========================
NUM_GPUS=2
CONFIG="DiTDH-S-synth_m.yaml"

IMAGE_PATH="/playpen/haochenz/LitsDatasets/128_len_img_one_per_channel/synth_m/train"
RESULTS_DIR="/playpen/haochenz/ckpts_diffusion_prior_0306/synth_m"
NUM_CH=2
export HF_HOME=/playpen/haochenz/hf_cache

PRECISION="bf16"   # fp32 / fp16 / bf16
WANDB=0            # 1 enable, 0 disable
COMPILE=1          # 1 enable, 0 disable
CKPT=""            # optional resume checkpoint
GLOBAL_SEED=0

# ==========================
# Env
# ==========================
export CUDA_VISIBLE_DEVICES=0,1

# reduce tokenizer parallel warning
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_NAME="DiTDH-S"
export ENTITY="zhanghaochen"
export PROJECT="Step2-Diffusion-Prior"
# optional debug flags
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ==========================
# Build command
# ==========================



CMD="torchrun --master_port=29634 --nproc_per_node=${NUM_GPUS} train.py \
  --config ${CONFIG} \
  --image-path ${IMAGE_PATH} \
  --num-ch ${NUM_CH} \
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