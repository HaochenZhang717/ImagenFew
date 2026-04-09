#!/usr/bin/env bash
#SBATCH --job-name=imagen_few
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/ECG200.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/ETTm1.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/ETTm2.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/ILI.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/SaugeenRiverFlow.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/SelfRegulationSCP1.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/Sine.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/StarLightCurves.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/Weather.yaml
