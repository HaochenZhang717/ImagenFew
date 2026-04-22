#!/usr/bin/env bash
#SBATCH --job-name=capcmp_ettm1_ve
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  WORK_DIR="$SLURM_SUBMIT_DIR"
else
  WORK_DIR="$SCRIPT_DIR"
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

# Defaults for your requested command; all can be overridden via sbatch --export.
DATASET="${DATASET:-ettm1_ve}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
CHECKPOINT="${CHECKPOINT:-/playpen-shared/haochenz/ImagenFew/caption_generator/logs/caption_generator/ettm1_stage1_qwen25_3b_ve/joint_caption_best.pt}"

echo "Running train compare on host $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "WORK_DIR=$WORK_DIR"
echo "DATASET=$DATASET"
echo "SPLIT=$SPLIT"
echo "MAX_SAMPLES=$MAX_SAMPLES"
echo "CHECKPOINT=$CHECKPOINT"

CMD=(
  env
  "DATASET=$DATASET"
  "SPLIT=$SPLIT"
  "MAX_SAMPLES=$MAX_SAMPLES"
  "CHECKPOINT=$CHECKPOINT"
  bash "$WORK_DIR/generate_caption_train_compare.sh"
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'FINAL_CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

