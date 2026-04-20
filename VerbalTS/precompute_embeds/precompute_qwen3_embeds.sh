#!/bin/bash

set -euo pipefail

CAPS_PATH="${CAPS_PATH:-../../data/VerbalTSDatasets/synthetic_u}"
SAVE_ROOT="${SAVE_ROOT:-$CAPS_PATH}"
SCRIPT_PATH="${SCRIPT_PATH:-precompute_qwen3_embeds.py}"
NPY_NAME="${NPY_NAME:-text_caps}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
USE_INSTRUCT="${USE_INSTRUCT:-0}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
USE_FP16="${USE_FP16:-0}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-qwen3_4b}"

mkdir -p "$SAVE_ROOT"

echo "Start precomputing Qwen3 embeddings..."
echo "CAPS_PATH=$CAPS_PATH"
echo "SAVE_ROOT=$SAVE_ROOT"
echo "MODEL_NAME=$MODEL_NAME"

run_split() {
    local split="$1"
    local out_path="$SAVE_ROOT/${split}_embeds_${OUTPUT_SUFFIX}.pt"
    local src_path="$CAPS_PATH/${split}_${NPY_NAME}.npy"

    if [ ! -f "$src_path" ]; then
        echo "Skip $split: $src_path not found"
        return 0
    fi

    local cmd=(
        python "$SCRIPT_PATH"
        --caps_path "$CAPS_PATH"
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

run_split train
run_split valid
run_split test
run_split generated

echo "Done!"
