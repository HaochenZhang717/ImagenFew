#!/usr/bin/env bash
#SBATCH --job-name=dp_base
#SBATCH --partition=blackwell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

echo "Running diffusion prior base on host $(hostname)"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
if [[ "$NPROC_PER_NODE" == *"("* ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE%%(*}"
fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$ROOT_DIR/diffusion_prior/train_diffusion_prior.py" \
  --config "$ROOT_DIR/diffusion_prior/configs/dit1d_base.yaml" \
  "$@"
