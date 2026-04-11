#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME=/playpen/haochenz/hf_cache

IMAGE_ROOT="${IMAGE_ROOT:-./logs/finetune_dataset_images}"
SAVE_ROOT="${SAVE_ROOT:-./logs/finetune_captions}"
SPLIT="${SPLIT:-train}"
LOG_ROOT="${LOG_ROOT:-./logs/finetune_caption_logs}"
GPU_IDS=(${GPU_IDS:-0 1 2 3})
NUM_PARTS="${NUM_PARTS:-${#GPU_IDS[@]}}"

DATASETS=(
  "AirQuality"
  "ETTh2"
)

if [[ "$NUM_PARTS" -ne "${#GPU_IDS[@]}" ]]; then
  echo "NUM_PARTS ($NUM_PARTS) must match number of GPU_IDS (${#GPU_IDS[@]})." >&2
  exit 1
fi

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

for dataset in "${DATASETS[@]}"; do
  image_folder="$IMAGE_ROOT/$dataset/$SPLIT"
  save_dir="$SAVE_ROOT/$dataset"
  dataset_log_dir="$LOG_ROOT/$dataset"

  if [[ ! -d "$image_folder" ]]; then
    echo "Image folder not found, skipping: $image_folder" >&2
    continue
  fi

  mkdir -p "$save_dir" "$dataset_log_dir"
  pids=()

  for ((part_id=0; part_id<NUM_PARTS; part_id++)); do
    gpu_id="${GPU_IDS[$part_id]}"
    log_file="$dataset_log_dir/${SPLIT}_part${part_id}_of_${NUM_PARTS}_gpu${gpu_id}.log"
    echo "Running captions for $dataset split=$SPLIT part ${part_id}/${NUM_PARTS} on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/run_caption.py \
      --part_id "$part_id" \
      --num_parts "$NUM_PARTS" \
      --image_folder "$image_folder" \
      --split "$SPLIT" \
      --dataset_name "$dataset" \
      --save_dir "$save_dir" >"$log_file" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
