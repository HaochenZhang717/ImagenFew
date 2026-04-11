#!/bin/bash

# ==========================
# HuggingFace cache
# ==========================
export HF_HOME=/playpen/haochenz/hf_cache
#export HF_HOME=/playpen-shared/haochenz/hf_cache

# ==========================
# Basic config
# ==========================
#CONFIG="/playpen-shared/haochenz/ckpts_diffusion_prior/DiTDH-S/config.yaml"
#CKPT="/playpen-shared/haochenz/ckpts_diffusion_prior/DiTDH-S/checkpoints/ep-last.pt"
CONFIG="/playpen/haochenz/ckpts_diffusion_prior_0301/DiTDH-S/config.yaml"
CKPT="/playpen/haochenz/ckpts_diffusion_prior_0301/DiTDH-S/checkpoints/ep-last.pt"

#scp -r haochenz@unites4.cs.unc.edu:/playpen-shared/haochenz/ckpts_diffusion_prior/DiTDH-S  ./
#OUTPUT_DIR="/playpen-shared/haochenz/diffusion_prior_results"
OUTPUT_DIR="/playpen/haochenz/diffusion_prior_results_0301"
OUTPUT_JSONL="${OUTPUT_DIR}/DiTDH-S-samples.jsonl"
OUTPUT_ARRAY="${OUTPUT_DIR}/DiTDH-S-samples.npy"

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