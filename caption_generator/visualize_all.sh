#!/usr/bin/env bash
#SBATCH --job-name=caption_visualize
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  WORK_DIR="$SLURM_SUBMIT_DIR"
else
  WORK_DIR="$(pwd)"
fi

if [[ -d "$WORK_DIR/caption_generator" && -d "$WORK_DIR/scripts" ]]; then
  ROOT_DIR="$WORK_DIR"
elif [[ "$(basename "$WORK_DIR")" == "caption_generator" && -d "$WORK_DIR/../scripts" ]]; then
  ROOT_DIR="$(cd "$WORK_DIR/.." && pwd)"
else
  echo "Could not infer project root from WORK_DIR=$WORK_DIR" >&2
  exit 1
fi

cd "$WORK_DIR"
mkdir -p /playpen-shared/haochenz/logs/slurm

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BIN=""
  if [[ -x "/playpen/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen/haochenz/miniconda3/bin/conda"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  else
    echo "Could not find a usable conda binary." >&2
    exit 1
  fi
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER}}"
mkdir -p "$MPLCONFIGDIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-Embedding-4B}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
USE_INSTRUCT="${USE_INSTRUCT:-0}"
SAVE_EMBEDDINGS="${SAVE_EMBEDDINGS:-0}"

DEFAULT_DATASETS=(
  "synthetic_m"
  "istanbul_traffic"
  "ETTm1"
#  "synthetic_u"
)

if [[ -n "${DATASET:-}" ]]; then
  DATASETS=("$DATASET")
elif [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASETS <<<"$DATASETS"
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

echo "Running caption embedding visualization on host $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<managed by slurm>}"
echo "MODEL_NAME=$MODEL_NAME"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "DEVICE=$DEVICE"
echo "DATASETS:"
printf '  %s\n' "${DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
  generated_path="../data/VerbalTSDatasets/${dataset}/generated_text_caps.npy"
  real_path="../data/VerbalTSDatasets/${dataset}/test_text_caps.npy"
  output_dir="./visuals/${dataset}"
  mkdir -p output_dir
  CMD=(
    python "visualize_caption_embedding_distribution.py"
    --generated-path "$generated_path"
    --real-path "$real_path"
    --output-dir "$output_dir"
    --model-name "$MODEL_NAME"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
  )

  if [[ "$USE_INSTRUCT" == "1" ]]; then
    CMD+=(--use-instruct)
  fi

  if [[ "$SAVE_EMBEDDINGS" == "1" ]]; then
    CMD+=(--save-embeddings)
  fi

  if [[ "$#" -gt 0 ]]; then
    CMD+=("$@")
  fi

  printf 'FINAL_CMD:'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  "${CMD[@]}"
done
