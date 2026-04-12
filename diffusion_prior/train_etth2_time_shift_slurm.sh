#!/usr/bin/env bash
#SBATCH --job-name=dp_etth2_tds
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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

TIME_DIST_SHIFT="${TIME_DIST_SHIFT:?Please set TIME_DIST_SHIFT, e.g. TIME_DIST_SHIFT=1.0}"
SHIFT_TAG="${TIME_DIST_SHIFT//./p}"

LATENTS_PATH="${LATENTS_PATH:-/playpen-shared/haochenz/ImagenFew/logs/vae_latents/finetune_dataset/ETTh2_mu.pt}"
BASE_CKPT="${BASE_CKPT:-/playpen-shared/haochenz/ImagenFew/logs/diffusion_prior/dit1d_base/diffusion_prior_best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/playpen-shared/haochenz/ImagenFew/logs/diffusion_prior/time_shift_sweep/ETTh2}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/tds_${SHIFT_TAG}}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/diffusion_prior/configs/dit1d_base_finetune_etth2.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-diffusion-prior}"
WANDB_NAME="${WANDB_NAME:-dit1d_base_finetune_etth2_tds_${SHIFT_TAG}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

echo "Running ETTh2 diffusion prior finetune sweep on host $(hostname)"
echo "TIME_DIST_SHIFT=$TIME_DIST_SHIFT"
echo "LATENTS_PATH=$LATENTS_PATH"
echo "BASE_CKPT=$BASE_CKPT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "WANDB_NAME=$WANDB_NAME"

python "$ROOT_DIR/diffusion_prior/train_diffusion_prior.py" \
  --config "$CONFIG_PATH" \
  --latents "$LATENTS_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --finetune-ckpt "$BASE_CKPT" \
  --time-dist-shift "$TIME_DIST_SHIFT" \
  --wandb-name "$WANDB_NAME" \
  "$@"
