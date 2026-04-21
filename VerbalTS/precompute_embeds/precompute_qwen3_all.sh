#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_PATH:-$SCRIPT_DIR/precompute_qwen3_embeds.py}"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../../data/VerbalTSDatasets}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
USE_INSTRUCT="${USE_INSTRUCT:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
USE_FP16="${USE_FP16:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-qwen3_4b}"
NPY_NAME="${NPY_NAME:-text_caps}"

DATASETS=(
  "synthetic_u"
  "synthetic_m"
  "istanbul_traffic"
  "ETTm1"
#  "Weather"
#  "BlindWays"
)

run_split() {
    local dataset_dir="$1"
    local split="$2"
    local out_path="$dataset_dir/${split}_embeds_${OUTPUT_SUFFIX}.pt"
    local src_path="$dataset_dir/${split}_${NPY_NAME}.npy"

    if [ ! -f "$src_path" ]; then
        echo "Skip $split for $dataset_dir: $src_path not found"
        return 0
    fi

    local cmd=(
        python "$SCRIPT_PATH"
        --caps_path "$dataset_dir"
        --save_path "$out_path"
        --npy_name "$NPY_NAME"
        --split "$split"
        --batch_size "$BATCH_SIZE"
        --device "$DEVICE"
        --model_name "$MODEL_NAME"
        --max_length "$MAX_LENGTH"
    )

    if [ "$USE_INSTRUCT" = "1" ]; then
        cmd+=(--use_instruct)
    fi
    if [ "$USE_FLASH_ATTN" = "1" ]; then
        cmd+=(--use_flash_attn)
    fi
    if [ "$USE_FP16" = "1" ]; then
        cmd+=(--use_fp16)
    fi

    "${cmd[@]}"
}

echo "Start precomputing Qwen3 embeddings for all VerbalTS datasets..."
echo "DATA_ROOT=$DATA_ROOT"
echo "MODEL_NAME=$MODEL_NAME"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "DEVICE=$DEVICE"

for dataset in "${DATASETS[@]}"; do
    dataset_dir="$DATA_ROOT/$dataset"
    if [ ! -d "$dataset_dir" ]; then
        echo "Skip dataset $dataset: $dataset_dir not found"
        continue
    fi

    echo "=============================="
    echo "Dataset: $dataset"
    echo "Directory: $dataset_dir"
    echo "=============================="

    run_split "$dataset_dir" "train"
    run_split "$dataset_dir" "valid"
    run_split "$dataset_dir" "test"
    run_split "$dataset_dir" "generated"
done

echo "Done!"
