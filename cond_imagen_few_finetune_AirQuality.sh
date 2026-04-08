#!/usr/bin/env bash
#SBATCH --job-name=cond_airquality
#SBATCH --partition=blackwell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

echo "Running conditional ImagenFew finetune AirQuality on host $(hostname)"
python -u "$ROOT_DIR/run.py" \
  --subset_p 1.0 \
  --wandb \
  --wandb_project CondImagenFewFinetune \
  --config "$ROOT_DIR/configs/conditional_imagen_few/AirQuality.yaml" \
  "$@"
