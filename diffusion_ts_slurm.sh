#!/usr/bin/env bash
#SBATCH --job-name=diffusion_ts
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if ! command -v conda >/dev/null 2>&1; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  elif [[ -f "/playpen/haochenz/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "/playpen/haochenz/miniconda3/etc/profile.d/conda.sh"
  fi
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/ETTm1.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/ETTm2.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/ILI.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/Mujoco.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/SaugeenRiverFlow.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/SelfRegulationSCP1.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/Sine.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/StarLightCurves.yaml

python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project DiffusionTS \
--config ./configs/DiffusionTS/Weather.yaml
