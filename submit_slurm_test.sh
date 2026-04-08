#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/slurm"
mkdir -p "$LOG_DIR"

PARTITION="${PARTITION:-all}"
ACCOUNT_ARGS=()
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  ACCOUNT_ARGS+=(--account "$SLURM_ACCOUNT")
fi

JOB_ID=$(sbatch --parsable --partition "$PARTITION" "${ACCOUNT_ARGS[@]}" "$ROOT_DIR/slurm_test_job.sh")

echo "Submitted Slurm test job: $JOB_ID"
echo "Partition: $PARTITION"
echo "Check status with: squeue -j $JOB_ID"
echo "Logs will appear under $LOG_DIR"
