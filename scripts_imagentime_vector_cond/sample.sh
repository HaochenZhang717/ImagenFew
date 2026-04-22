#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_ROOT="${LOG_ROOT:-/playpen-shared/haochenz/ImagenFew/logs/ImagenTimeVectorCond}"
DATA_ROOT="${DATA_ROOT:-/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/playpen-shared/haochenz/ImagenFew/scripts_imagentime_vector_cond}"
METRIC_ITERATION="${METRIC_ITERATION:-10}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-train}"
SEED="${SEED:-0}"
EVAL_METRICS="${EVAL_METRICS:-disc vaeFID}"
FID_VAE_CKPT_ROOT="${FID_VAE_CKPT_ROOT:-/playpen-shared/haochenz/ImagenFew/fid_vae_ckpts}"
TS2VEC_ROOT="${TS2VEC_ROOT:-$OUTPUT_ROOT/TS2VEC}"
RUN_SUFFIX="${RUN_SUFFIX:-gs1p0_ld0p0_sp1p0}"

DATASETS=(
  "ETTm1"
  "synthetic_u"
  "synthetic_m"
  "istanbul_traffic"
)

declare -A CONFIG_MAP=(
  ["ETTm1"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_ETTm1_qwen3.yaml"
  ["synthetic_u"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_synthetic_u_qwen3.yaml"
  ["synthetic_m"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_synthetic_m_qwen3.yaml"
  ["istanbul_traffic"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_istanbul_traffic_qwen3.yaml"
)

declare -A RUN_DIR_MAP=(
  ["ETTm1"]="$LOG_ROOT/VerbalTS_ETTm1_qwen3/ETTm1_qwen3_${RUN_SUFFIX}"
  ["synthetic_u"]="$LOG_ROOT/VerbalTS_synthetic_u_qwen3/synthetic_u_qwen3_${RUN_SUFFIX}"
  ["synthetic_m"]="$LOG_ROOT/VerbalTS_synthetic_m_qwen3/synthetic_m_qwen3_${RUN_SUFFIX}"
  ["istanbul_traffic"]="$LOG_ROOT/VerbalTS_istanbul_traffic_qwen3/istanbul_traffic_qwen3_${RUN_SUFFIX}"
)

mkdir -p "$OUTPUT_ROOT" "$TS2VEC_ROOT"

for dataset in "${DATASETS[@]}"; do
  config_path="${CONFIG_MAP[$dataset]}"
  run_dir="${RUN_DIR_MAP[$dataset]}"
  ckpt_dir="$run_dir/checkpoints"
  embeds_path="$DATA_ROOT/$dataset/generated_embeds_qwen3_4b.pt"
  samples_dir="$run_dir/eval_samples_generated_qwen3"
  jsonl_path="$OUTPUT_ROOT/${dataset}_generated_qwen3_metrics.jsonl"
  ts2vec_dir="$TS2VEC_ROOT/$dataset"

  if [[ ! -f "$config_path" ]]; then
    echo "Missing config: $config_path" >&2
    exit 1
  fi
  if [[ ! -d "$ckpt_dir" ]]; then
    echo "Missing checkpoint dir: $ckpt_dir" >&2
    exit 1
  fi
  if [[ ! -f "$embeds_path" ]]; then
    echo "Missing embeddings: $embeds_path" >&2
    exit 1
  fi

  mkdir -p "$samples_dir" "$ts2vec_dir"
  : > "$jsonl_path"

  echo "=== Dataset: $dataset ==="
  echo "Config: $config_path"
  echo "Checkpoint dir: $ckpt_dir"
  echo "Embeddings: $embeds_path"
  echo "Metrics jsonl: $jsonl_path"

  mapfile -t ckpts < <(find "$ckpt_dir" -maxdepth 1 -type f -name 'epoch_*.pt' | sort -V)
  if [[ "${#ckpts[@]}" -eq 0 ]]; then
    echo "No checkpoints found under $ckpt_dir" >&2
    exit 1
  fi

  for ckpt_path in "${ckpts[@]}"; do
    ckpt_tag="$(basename "$ckpt_path" .pt)"
    output_path="$samples_dir/${dataset}_generated_embeds_qwen3_4b_${ckpt_tag}.pt"

    echo "--- Sampling $dataset / $ckpt_tag ---"
    "$PYTHON_BIN" "$ROOT_DIR/scripts_imagentime_vector_cond/sample_imagen_time_vectorcond.py" \
      --config "$config_path" \
      --dataset "$dataset" \
      --model-ckpt "$ckpt_path" \
      --embeds-path "$embeds_path" \
      --reference-split "$REFERENCE_SPLIT" \
      --seed "$SEED" \
      --metric-iteration "$METRIC_ITERATION" \
      --fid-vae-ckpt-root "$FID_VAE_CKPT_ROOT" \
      --ts2vec-dir "$ts2vec_dir" \
      --output "$output_path" \
      --output-jsonl "$jsonl_path" \
      --eval-metrics $EVAL_METRICS
  done
done
