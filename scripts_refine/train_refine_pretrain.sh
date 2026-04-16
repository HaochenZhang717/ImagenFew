#!/usr/bin/env bash
#SBATCH --job-name=refine_pretrain
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

CONFIG="${CONFIG:-$ROOT_DIR/configs/refine/pretrain.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-ImagenFewRefine}"
SUBSET_P="${SUBSET_P:-1.0}"

echo "Running ImagenFewRefine pretrain on host $(hostname)"
echo "CONFIG=${CONFIG}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"

python -u "$ROOT_DIR/run.py" \
  --subset_p "$SUBSET_P" \
  --wandb \
  --wandb_project "$WANDB_PROJECT" \
  --config "$CONFIG" \
  --no_test_model \
  "$@"
