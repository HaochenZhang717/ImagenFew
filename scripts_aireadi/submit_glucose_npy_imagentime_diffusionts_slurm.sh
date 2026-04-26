#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/logs/slurm"

PARTITION="${PARTITION:-a6000}"
SLURM_TIME="${SLURM_TIME:-2-00:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-60G}"
GPUS="${GPUS:-1}"
CONDA_ENV="${CONDA_ENV:-vlm}"
SUBSET_P="${SUBSET_P:-1.0}"
USE_WANDB="${USE_WANDB:-1}"

IMAGENTIME_CONFIG="${IMAGENTIME_CONFIG:-./configs/ImagenTime/Glucose.yaml}"
DIFFUSIONTS_CONFIG="${DIFFUSIONTS_CONFIG:-./configs/DiffusionTS/Glucose.yaml}"
IMAGENTIME_WANDB_PROJECT="${IMAGENTIME_WANDB_PROJECT:-ImagenTime}"
DIFFUSIONTS_WANDB_PROJECT="${DIFFUSIONTS_WANDB_PROJECT:-DiffusionTS}"

submit_job() {
  local job_name="$1"
  local command="$2"
  local script
  local wrapped

  script=$(cat <<EOF
cd "$ROOT_DIR"
source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "$CONDA_ENV" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
  else
    echo "CONDA_ENV is set to '$CONDA_ENV', but conda is not available in PATH." >&2
    exit 1
  fi
fi
$command
EOF
)
  printf -v wrapped '%q' "$script"

  sbatch --parsable \
    --job-name="$job_name" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS_PER_TASK" \
    --gres="gpu:${GPUS}" \
    --mem="$MEM" \
    --time="$SLURM_TIME" \
    --output="$ROOT_DIR/logs/slurm/%x_%j.out" \
    --error="$ROOT_DIR/logs/slurm/%x_%j.err" \
    --export=ALL,CONDA_ENV="$CONDA_ENV",SUBSET_P="$SUBSET_P",USE_WANDB="$USE_WANDB" \
    --wrap="bash -lc $wrapped"
}

IMAGENTIME_JOB=$(submit_job \
  "glucose_imagentime" \
  "CONFIG='$IMAGENTIME_CONFIG' WANDB_PROJECT='$IMAGENTIME_WANDB_PROJECT' bash scripts_aireadi/train_imagentime_glucose_npy.sh")

DIFFUSIONTS_JOB=$(submit_job \
  "glucose_diffusionts" \
  "CONFIG='$DIFFUSIONTS_CONFIG' WANDB_PROJECT='$DIFFUSIONTS_WANDB_PROJECT' bash scripts_aireadi/train_diffusionts_glucose_npy.sh")

echo "Submitted glucose ImagenTime job:   $IMAGENTIME_JOB"
echo "Submitted glucose DiffusionTS job:  $DIFFUSIONTS_JOB"
echo "Logs: $ROOT_DIR/logs/slurm"
