#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-vlm}"
SLURM_TIME="${SLURM_TIME:-2-00:00:00}"

CLEAN_CONFIG="${CLEAN_CONFIG:-./configs/self_cond_finetune/ETTh2_clean.yaml}"
DEGRADED_CONFIG="${DEGRADED_CONFIG:-./configs/self_cond_finetune/ETTh2_degraded_alpha0p1.yaml}"
PRIOR_REFINE_CONFIG="${PRIOR_REFINE_CONFIG:-./configs/self_cond_finetune/ETTh2_prior_refine_alpha0p1.yaml}"

echo "[INFO] Submitting clean posterior training with CONFIG=${CLEAN_CONFIG}"
CONDA_ENV="${CONDA_ENV}" \
CONFIG="${CLEAN_CONFIG}" \
sbatch --time="${SLURM_TIME}" self_cond_finetune_etth2_slurm.sh

echo "[INFO] Submitting degraded posterior training with CONFIG=${DEGRADED_CONFIG}"
CONDA_ENV="${CONDA_ENV}" \
CONFIG="${DEGRADED_CONFIG}" \
sbatch --time="${SLURM_TIME}" self_cond_finetune_etth2_slurm.sh

echo "[INFO] Submitting prior_refine training with CONFIG=${PRIOR_REFINE_CONFIG}"
CONDA_ENV="${CONDA_ENV}" \
CONFIG="${PRIOR_REFINE_CONFIG}" \
sbatch --time="${SLURM_TIME}" self_cond_finetune_etth2_slurm.sh
