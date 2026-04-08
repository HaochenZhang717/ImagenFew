#!/usr/bin/env bash
#SBATCH --job-name=slurm_test
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

echo "hostname: $(hostname)"
echo "date: $(date)"
echo "user: $USER"
echo "pwd: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

python - <<'PY'
import os
import socket
print("python_ok:", True)
print("host:", socket.gethostname())
print("cwd:", os.getcwd())
PY
