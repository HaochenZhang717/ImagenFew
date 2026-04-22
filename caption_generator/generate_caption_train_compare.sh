#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"
export TOKENIZERS_PARALLELISM=false

DATASET="${DATASET:-ettm1}"
SPLIT="${SPLIT:-train}"
OUTPUT_JSONL="${OUTPUT_JSONL:-}"
OUTPUT_SUMMARY="${OUTPUT_SUMMARY:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

case "${DATASET}" in
  ettm1|ETTm1)
    DATASET_NAME="ETTm1"
    CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/caption_generator/ettm1_stage1_qwen25_3b/joint_caption_best.pt}"
    CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/ettm1_stage1_qwen25_3b.yaml}"
    DEFAULT_OUTPUT_JSONL="../data/VerbalTSDatasets/ETTm1/${SPLIT}_generated_vs_gt.jsonl"
    ;;
  synthetic_m)
    DATASET_NAME="synthetic_m"
    CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/caption_generator/synthetic_m_stage1_qwen25_3b/joint_caption_best.pt}"
    CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/synthetic_m_stage1_qwen25_3b.yaml}"
    DEFAULT_OUTPUT_JSONL="../data/VerbalTSDatasets/synthetic_m/${SPLIT}_generated_vs_gt.jsonl"
    ;;
  synthetic_u)
    DATASET_NAME="synthetic_u"
    CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/caption_generator/synthetic_u_stage1_qwen25_3b/joint_caption_best.pt}"
    CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/synthetic_u_stage1_qwen25_3b.yaml}"
    DEFAULT_OUTPUT_JSONL="../data/VerbalTSDatasets/synthetic_u/${SPLIT}_generated_vs_gt.jsonl"
    ;;
  istanbul|istanbul_traffic)
    DATASET_NAME="istanbul_traffic"
    CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/caption_generator/istanbul_stage1_qwen25_3b/joint_caption_best.pt}"
    CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/istanbul_stage1_qwen25_3b.yaml}"
    DEFAULT_OUTPUT_JSONL="../data/VerbalTSDatasets/istanbul_traffic/${SPLIT}_generated_vs_gt.jsonl"
    ;;
  blindways|BlindWays)
    DATASET_NAME="BlindWays"
    CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/caption_generator/blindways_stage1_qwen25_3b/joint_caption_best.pt}"
    CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/blindways_stage1_qwen25_3b.yaml}"
    DEFAULT_OUTPUT_JSONL="../data/VerbalTSDatasets/BlindWays/${SPLIT}_generated_vs_gt.jsonl"
    ;;
  weather|Weather)
    DATASET_NAME="Weather"
    CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/logs/caption_generator/weather_stage1_qwen25_3b/joint_caption_best.pt}"
    CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/weather_stage1_qwen25_3b.yaml}"
    DEFAULT_OUTPUT_JSONL="../data/VerbalTSDatasets/Weather/${SPLIT}_generated_vs_gt.jsonl"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

if [[ -z "$OUTPUT_JSONL" ]]; then
  OUTPUT_JSONL="$DEFAULT_OUTPUT_JSONL"
fi

CMD=(
  python "$SCRIPT_DIR/generate_stage1_compare.py"
  --config "$CONFIG"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --output-jsonl "$OUTPUT_JSONL"
)

if [[ -n "$OUTPUT_SUMMARY" ]]; then
  CMD+=(--output-summary "$OUTPUT_SUMMARY")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(--max-samples "$MAX_SAMPLES")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

echo "Running Stage1 caption compare for dataset ${DATASET_NAME}"
printf 'FINAL_CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
