#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SLURM_SCRIPT="${SLURM_SCRIPT:-./train_imagentime_trend_slurm.sh}"

echo "Submitting ImagenTime trend jobs for ETTh2, AirQuality, and Mujoco"

echo "Submitting ETTh2 trend"
CONFIG=./configs/ImagenTimeDecomposed/ETTh2_trend.yaml sbatch "$SLURM_SCRIPT" "$@"

echo "Submitting AirQuality trend"
CONFIG=./configs/ImagenTimeDecomposed/AirQuality_trend.yaml sbatch "$SLURM_SCRIPT" "$@"

echo "Submitting Mujoco trend"
CONFIG=./configs/ImagenTimeDecomposed/Mujoco_trend.yaml sbatch "$SLURM_SCRIPT" "$@"
