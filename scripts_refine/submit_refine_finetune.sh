#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_VALUE="${CONDA_ENV:-vlm}"
SLURM_TIME_VALUE="${SLURM_TIME:-2-00:00:00}"
ETTH2_CONFIG="${ETTH2_CONFIG:-$ROOT_DIR/configs/refine/ETTh2.yaml}"
AIRQUALITY_CONFIG="${AIRQUALITY_CONFIG:-$ROOT_DIR/configs/refine/AirQuality.yaml}"
MUJOCO_CONFIG="${MUJOCO_CONFIG:-$ROOT_DIR/configs/refine/mujoco.yaml}"
WANDB_PROJECT_VALUE="${WANDB_PROJECT:-ImagenFewRefine}"

ETTH2_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" WANDB_PROJECT="$WANDB_PROJECT_VALUE" CONFIG="$ETTH2_CONFIG" sbatch --parsable --time="$SLURM_TIME_VALUE" "$ROOT_DIR/scripts_refine/train_refine_etth2.sh")
AIRQUALITY_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" WANDB_PROJECT="$WANDB_PROJECT_VALUE" CONFIG="$AIRQUALITY_CONFIG" sbatch --parsable --time="$SLURM_TIME_VALUE" "$ROOT_DIR/scripts_refine/train_refine_airquality.sh")
MUJOCO_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" WANDB_PROJECT="$WANDB_PROJECT_VALUE" CONFIG="$MUJOCO_CONFIG" sbatch --parsable --time="$SLURM_TIME_VALUE" "$ROOT_DIR/scripts_refine/train_refine_mujoco.sh")

echo "Submitted ETTh2 refine finetune job:      $ETTH2_JOB"
echo "Submitted AirQuality refine finetune job: $AIRQUALITY_JOB"
echo "Submitted Mujoco refine finetune job:     $MUJOCO_JOB"
