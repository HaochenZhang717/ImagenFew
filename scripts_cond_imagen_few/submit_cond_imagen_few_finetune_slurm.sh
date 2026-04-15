#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_VALUE="${CONDA_ENV:-vlm}"
SLURM_TIME_VALUE="${SLURM_TIME:-2-00:00:00}"

ETTH2_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" sbatch --parsable --time="$SLURM_TIME_VALUE" "$SCRIPT_DIR/cond_imagen_few_finetune_ETTh2.sh")
AIRQUALITY_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" sbatch --parsable --time="$SLURM_TIME_VALUE" "$SCRIPT_DIR/cond_imagen_few_finetune_AirQuality.sh")
MUJOCO_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" sbatch --parsable --time="$SLURM_TIME_VALUE" "$SCRIPT_DIR/cond_imagen_few_finetune_mujoco.sh")

echo "Submitted ETTh2 job:      $ETTH2_JOB"
echo "Submitted AirQuality job: $AIRQUALITY_JOB"
echo "Submitted Mujoco job:     $MUJOCO_JOB"
