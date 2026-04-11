#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python scripts/export_finetune_datasets_npy.py \
  --dataset-config-dir ./configs/self_cond_finetune \
  --output-dir ./logs/finetune_datasets_npy \
  --split train \
  --batch-size 256
