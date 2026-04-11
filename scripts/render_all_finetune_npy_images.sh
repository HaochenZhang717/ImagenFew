#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NPY_DIR="${NPY_DIR:-./logs/finetune_datasets_npy}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs/finetune_dataset_images}"
NUM_SEGMENTS="${NUM_SEGMENTS:-1}"
SEGMENT_LEN="${SEGMENT_LEN:-}"
WORKERS="${WORKERS:-}"

for ts_path in "$NPY_DIR"/*_train.npy; do
  dataset_name="$(basename "$ts_path" _train.npy)"
  save_dir="$OUTPUT_DIR/$dataset_name/train"
  echo "Rendering $dataset_name from $ts_path to $save_dir"

  cmd=(
    python scripts/render_finetune_npy_images.py
    --ts-path "$ts_path"
    --save-dir "$save_dir"
    --num-segments "$NUM_SEGMENTS"
  )

  if [[ -n "$SEGMENT_LEN" ]]; then
    cmd+=(--segment-len "$SEGMENT_LEN")
  fi

  if [[ -n "$WORKERS" ]]; then
    cmd+=(--workers "$WORKERS")
  fi

  "${cmd[@]}"
done
