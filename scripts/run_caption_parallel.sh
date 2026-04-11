#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-/playpen/haochenz/hf_cache}"

IMAGE_ROOT="${IMAGE_ROOT:-./logs/finetune_dataset_images}"
SAVE_ROOT="${SAVE_ROOT:-./logs/finetune_captions}"
LOG_ROOT="${LOG_ROOT:-./logs/finetune_caption_logs}"
SPLIT="${SPLIT:-train}"
GPU_IDS=(${GPU_IDS:-0 1 2 3})
NUM_PARTS="${NUM_PARTS:-${#GPU_IDS[@]}}"
QUIET="${QUIET:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DO_SAMPLE="${DO_SAMPLE:-0}"

if [[ "$NUM_PARTS" -lt 1 ]]; then
  echo "NUM_PARTS must be >= 1" >&2
  exit 1
fi

mkdir -p "$SAVE_ROOT" "$LOG_ROOT"

mapfile -t DATASETS < <(
  find "$IMAGE_ROOT" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort
)

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "No dataset folders found under $IMAGE_ROOT" >&2
  exit 1
fi

TASKS=()
for dataset in "${DATASETS[@]}"; do
  image_folder="$IMAGE_ROOT/$dataset/$SPLIT"
  if [[ ! -d "$image_folder" ]]; then
    echo "Image folder not found, skipping: $image_folder" >&2
    continue
  fi
  for ((part_id=0; part_id<NUM_PARTS; part_id++)); do
    TASKS+=("${dataset}:${part_id}")
  done
done

if [[ "${#TASKS[@]}" -eq 0 ]]; then
  echo "No caption tasks found for split=$SPLIT under $IMAGE_ROOT" >&2
  exit 1
fi

for ((start=0; start<${#TASKS[@]}; start+=${#GPU_IDS[@]})); do
  pids=()
  for ((gpu_slot=0; gpu_slot<${#GPU_IDS[@]}; gpu_slot++)); do
    task_idx=$((start + gpu_slot))
    if (( task_idx >= ${#TASKS[@]} )); then
      break
    fi

    gpu_id="${GPU_IDS[$gpu_slot]}"
    task="${TASKS[$task_idx]}"
    dataset="${task%%:*}"
    part_id="${task##*:}"

    image_folder="$IMAGE_ROOT/$dataset/$SPLIT"
    save_dir="$SAVE_ROOT/$dataset"
    dataset_log_dir="$LOG_ROOT/$dataset"
    log_file="$dataset_log_dir/${SPLIT}_part${part_id}_of_${NUM_PARTS}_gpu${gpu_id}.log"

    mkdir -p "$save_dir" "$dataset_log_dir"
    echo "Running captions for $dataset split=$SPLIT part ${part_id}/${NUM_PARTS} on GPU $gpu_id"

    cmd=(
      python scripts/run_caption.py
      --part_id "$part_id"
      --num_parts "$NUM_PARTS"
      --image_folder "$image_folder"
      --split "$SPLIT"
      --dataset_name "$dataset"
      --save_dir "$save_dir"
      --batch-size "$BATCH_SIZE"
      --max-new-tokens "$MAX_NEW_TOKENS"
    )
    if [[ "$DO_SAMPLE" == "1" ]]; then
      cmd+=(--do-sample)
    fi
    if [[ "$QUIET" == "1" ]]; then
      cmd+=(--quiet)
    fi

    CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" >"$log_file" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done
