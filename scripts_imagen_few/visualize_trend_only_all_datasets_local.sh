#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-train}"
SAMPLE_IDX="${SAMPLE_IDX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/logs/trend_only_debug}"

mkdir -p "${OUTPUT_DIR}"

run_one() {
  local dataset_name="$1"
  local config_path="$2"
  local lower_name="$3"

  echo "[INFO] Visualizing ${dataset_name} (${SPLIT}, sample ${SAMPLE_IDX})"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts_imagen_few/visualize_trend_only_preprocessing.py" \
    --config "${config_path}" \
    --dataset "${dataset_name}" \
    --split "${SPLIT}" \
    --sample-idx "${SAMPLE_IDX}" \
    --all-channels \
    --save "${OUTPUT_DIR}/${lower_name}_${SPLIT}_sample${SAMPLE_IDX}_all_channels.png" \
    --stats-json "${OUTPUT_DIR}/${lower_name}_${SPLIT}_sample${SAMPLE_IDX}_all_channels.json"
}

run_one "ETTh2" "${ROOT_DIR}/configs/finetune/ETTh2.yaml" "etth2"
run_one "AirQuality" "${ROOT_DIR}/configs/finetune/AirQuality.yaml" "airquality"
run_one "mujoco" "${ROOT_DIR}/configs/finetune/Mujoco.yaml" "mujoco"

echo "[INFO] Done. Outputs saved under ${OUTPUT_DIR}"
