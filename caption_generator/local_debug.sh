#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

torchrun \
  --standalone \
  --nproc_per_node=4 \
  train_stage1.py \
  --config configs/ettm1_stage1_qwen25_3b.yaml \
  --override training.ddp=true
