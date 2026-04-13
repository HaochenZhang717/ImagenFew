#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-vlm}"
SLURM_TIME="${SLURM_TIME:-2-00:00:00}"

echo "[INFO] Submitting diffusion prior finetune jobs with CONDA_ENV=${CONDA_ENV}"

#echo "[INFO] Submitting ETTh2 finetune"
#CONDA_ENV="${CONDA_ENV}" \
#sbatch --time="${SLURM_TIME}" diffusion_prior/train_etth2_finetune_slurm.sh

echo "[INFO] Submitting mujoco finetune"
CONDA_ENV="${CONDA_ENV}" \
sbatch --time="${SLURM_TIME}" diffusion_prior/train_mujoco_finetune_slurm.sh

echo "[INFO] Submitting AirQuality finetune"
CONDA_ENV="${CONDA_ENV}" \
sbatch --time="${SLURM_TIME}" diffusion_prior/train_airquality_finetune_slurm.sh


#CONDA_ENV=vlm bash diffusion_prior/submit_diffusion_prior_finetune_slurm.sh
