#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_VALUE="${CONDA_ENV:-vlm}"
SLURM_TIME_VALUE="${SLURM_TIME:-2-00:00:00}"
SUBSET_P="${SUBSET_P:-0.001}"

IMAGENTIME_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" SUBSET_P="$SUBSET_P" sbatch --parsable --time="$SLURM_TIME_VALUE" "$ROOT_DIR/scripts_aireadi/train_imagentime_aireadi_slurm.sh")
DIFFUSIONTS_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" SUBSET_P="$SUBSET_P" sbatch --parsable --time="$SLURM_TIME_VALUE" "$ROOT_DIR/scripts_aireadi/train_diffusionts_aireadi_slurm.sh")

echo "Submitted AIREADI ImagenTime job:   $IMAGENTIME_JOB"
echo "Submitted AIREADI DiffusionTS job:  $DIFFUSIONTS_JOB"
