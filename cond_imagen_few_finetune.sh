#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/slurm"
mkdir -p "$LOG_DIR"

PARTITION="${PARTITION:-blackwell}"
ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS+=(--account "$SLURM_ACCOUNT")
fi

ETTH2_JOB=$(sbatch --parsable --partition "$PARTITION" "${ACCOUNT_ARGS[@]}" "$ROOT_DIR/cond_imagen_few_finetune_ETTh2.sh")
MUJOCO_JOB=$(sbatch --parsable --partition "$PARTITION" "${ACCOUNT_ARGS[@]}" "$ROOT_DIR/cond_imagen_few_finetune_mujoco.sh")
AIRQUALITY_JOB=$(sbatch --parsable --partition "$PARTITION" "${ACCOUNT_ARGS[@]}" "$ROOT_DIR/cond_imagen_few_finetune_AirQuality.sh")

echo "Submitted ETTh2 job:      $ETTH2_JOB"
echo "Submitted mujoco job:     $MUJOCO_JOB"
echo "Submitted AirQuality job: $AIRQUALITY_JOB"
echo "Partition: $PARTITION"
echo "Logs will appear under $LOG_DIR"
echo "Check status with: squeue -j $ETTH2_JOB,$MUJOCO_JOB,$AIRQUALITY_JOB"
