#!/usr/bin/env bash
#SBATCH --job-name=imagen_few_airquality
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"
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

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO


CONFIG="/playpen-shared/haochenz/ImagenFew/configs/refine/AirQuality.yaml"
WANDB_PROJECT=ImagenFewRefine
SUBSET_P="${SUBSET_P:-1.0}"

echo "Running ImagenFewRefine pretrain on host $(hostname)"
echo "CONFIG=${CONFIG}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"

pip install loralib
python -u "/playpen-shared/haochenz/ImagenFew/run_refine.py" \
  --subset_p "$SUBSET_P" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --config "$CONFIG" \
  "$@"
