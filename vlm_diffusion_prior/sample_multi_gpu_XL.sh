#!/bin/bash

# ==========================
# HuggingFace cache
# ==========================
export HF_HOME=/playpen/haochenz/hf_cache

# ==========================
# Basic config
# ==========================
CONFIG="/playpen-shared/haochenz/ckpts_diffusion_prior/DiTDH-XL/config.yaml"
CKPT="/playpen-shared/haochenz/ckpts_diffusion_prior/DiTDH-XL/checkpoints/ep-last.pt"

OUTPUT_DIR="/playpen/haochenz/diffusion_prior_results"
OUTPUT_JSONL="${OUTPUT_DIR}/DiTDH-XL-samples.jsonl"
OUTPUT_ARRAY="${OUTPUT_DIR}/DiTDH-XL-samples.npy"

# ==========================
# GPU settings
# ==========================
NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ==========================
# Make sure output directory exists
# ==========================
mkdir -p ${OUTPUT_DIR}

# ==========================
# Run multi-GPU sampling
# ==========================
torchrun --nproc_per_node=${NUM_GPUS} sample_multi_gpu.py \
  --config ${CONFIG} \
  --seed 42 \
  --num_samples 24000 \
  --batch_size 64 \
  --ckpt ${CKPT} \
  --output_jsonl ${OUTPUT_JSONL} \
  --output_array ${OUTPUT_ARRAY}