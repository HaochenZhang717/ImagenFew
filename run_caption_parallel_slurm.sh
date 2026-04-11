#!/usr/bin/env bash
#SBATCH --job-name=caption_parallel
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"
mkdir -p /playpen-shared/haochenz/logs/slurm

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

export HF_HOME="${HF_HOME:-/playpen/haochenz/hf_cache}"
export GPU_IDS="${GPU_IDS:-0 1 2 3}"
export NUM_PARTS="${NUM_PARTS:-4}"
export SPLIT="${SPLIT:-train}"
export IMAGE_ROOT="${IMAGE_ROOT:-./logs/finetune_dataset_images}"
export SAVE_ROOT="${SAVE_ROOT:-./logs/finetune_captions}"
export LOG_ROOT="${LOG_ROOT:-./logs/finetune_caption_logs}"
export QUIET="${QUIET:-1}"
export BATCH_SIZE="${BATCH_SIZE:-4}"

bash scripts/run_caption_parallel.sh
