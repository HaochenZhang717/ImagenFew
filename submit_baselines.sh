#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-vlm}"

CONDA_ENV="$CONDA_ENV" sbatch diffusion_ts_slurm.sh
CONDA_ENV="$CONDA_ENV" sbatch imagen_few_slurm.sh
CONDA_ENV="$CONDA_ENV" sbatch imagen_time_slurm.sh
