#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/slurm"
mkdir -p "$LOG_DIR"

PARTITION="${PARTITION:-blackwell}"
ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS+=(--account "$SLURM_ACCOUNT")
fi

SMALL_GPUS="${SMALL_GPUS:-1}"
BASE_GPUS="${BASE_GPUS:-1}"
LARGE_GPUS="${LARGE_GPUS:-1}"

SMALL_JOB=$(sbatch --parsable --partition "$PARTITION" --gres "gpu:${SMALL_GPUS}" "${ACCOUNT_ARGS[@]}" "$ROOT_DIR/diffusion_prior/train_small.sh")
BASE_JOB=$(sbatch --parsable --partition "$PARTITION" --gres "gpu:${BASE_GPUS}" "${ACCOUNT_ARGS[@]}" "$ROOT_DIR/diffusion_prior/train_base.sh")
LARGE_JOB=$(sbatch --parsable --partition "$PARTITION" --gres "gpu:${LARGE_GPUS}" "${ACCOUNT_ARGS[@]}" "$ROOT_DIR/diffusion_prior/train_large.sh")

echo "Submitted small job: $SMALL_JOB"
echo "Submitted base job:  $BASE_JOB"
echo "Submitted large job: $LARGE_JOB"
echo "Partition: $PARTITION"
echo "GPUs per job: small=$SMALL_GPUS base=$BASE_GPUS large=$LARGE_GPUS"
echo "Logs will appear under $LOG_DIR"
echo "Check status with: squeue -j $SMALL_JOB,$BASE_JOB,$LARGE_JOB"
