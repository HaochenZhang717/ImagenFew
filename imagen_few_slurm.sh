#!/usr/bin/env bash
#SBATCH --job-name=imagen_few
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
mkdir -p "$ROOT_DIR/logs/slurm"

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


python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/AirQuality.yaml


python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/ETTh2.yaml


python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/Mujoco.yaml




#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/ECG200.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/ETTm1.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/ETTm2.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/ILI.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/SaugeenRiverFlow.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/SelfRegulationSCP1.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/Sine.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/StarLightCurves.yaml

#python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/Weather.yaml
