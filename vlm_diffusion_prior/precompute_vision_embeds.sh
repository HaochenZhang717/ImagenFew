#!/bin/bash
set -e

# ==========================
# User Config
# ==========================
NUM_GPUS=8
CONFIG="DiTDH-S.yaml"

#IMAGE_PATH="/playpen/haochenz/LitsDatasets/128_len_img/synth_u/train"
#JSONL_PATH="/playpen/haochenz/LitsDatasets/128_len_caps/synth_u/train_caps.jsonl"
#SAVE_PATH="/playpen/haochenz/LitsDatasets/128_len_vision_latent/synth_u/train.pt"

# ==========================
# Input args
# ==========================

if [ $# -lt 3 ]; then
  echo "Usage: $0 IMAGE_PATH NUM_CH SAVE_PATH"
  exit 1
fi

IMAGE_PATH=$1
NUM_CH=$2
SAVE_PATH=$3



RESULTS_DIR="none"

PRECISION="bf16"   # fp32 / fp16 / bf16
WANDB=0            # 1 enable, 0 disable
COMPILE=1          # 1 enable, 0 disable
CKPT=""            # optional resume checkpoint
GLOBAL_SEED=0

# ==========================
# Env
# ==========================
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# reduce tokenizer parallel warning
export TOKENIZERS_PARALLELISM=false
export EXPERIMENT_NAME="DiTDH-S"
export ENTITY="zhanghaochen"
export PROJECT="Step2-Diffusion-Prior"
export HF_HOME=/playpen/haochenz/hf_cache


# ==========================
# Build command
# ==========================
CMD="torchrun --nproc_per_node=${NUM_GPUS} precompute_vision_embeds.py \
  --config ${CONFIG} \
  --image-path ${IMAGE_PATH} \
  --num-ch ${NUM_CH} \
  --results-dir ${RESULTS_DIR} \
  --precision ${PRECISION} \
  --global-seed ${GLOBAL_SEED} \
  --save-path ${SAVE_PATH}"

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
