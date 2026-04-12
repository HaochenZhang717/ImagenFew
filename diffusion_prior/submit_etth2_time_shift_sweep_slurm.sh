#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIME_DIST_SHIFTS=(${TIME_DIST_SHIFTS:-1.0 2.0 5.0 10.0})
SCRIPT_PATH="${SCRIPT_PATH:-$ROOT_DIR/diffusion_prior/train_etth2_time_shift_slurm.sh}"

echo "Submitting ETTh2 time_dist_shift sweep with values: ${TIME_DIST_SHIFTS[*]}"

for shift in "${TIME_DIST_SHIFTS[@]}"; do
  shift_tag="${shift//./p}"
  job_name="dp_tds_${shift_tag}"
  echo "Submitting $job_name (time_dist_shift=$shift)"
  TIME_DIST_SHIFT="$shift" sbatch --job-name="$job_name" "$SCRIPT_PATH" "$@"
done
