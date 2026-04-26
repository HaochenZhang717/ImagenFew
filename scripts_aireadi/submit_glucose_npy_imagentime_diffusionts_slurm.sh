#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$ROOT_DIR/logs/slurm"

PARTITION="${PARTITION:-all}"
SLURM_TIME="${SLURM_TIME:-1-00:00:00}"
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
  local config="$2"
  local wandb_project="$3"
  local train_script="$4"

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
    --export=ALL <<EOF
#!/usr/bin/env bash
set -euo pipefail

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

CONFIG="$config" \
WANDB_PROJECT="$wandb_project" \
SUBSET_P="$SUBSET_P" \
USE_WANDB="$USE_WANDB" \
bash "$train_script"
EOF
}

IMAGENTIME_JOB=$(submit_job \
  "glucose_imagentime" \
  "$IMAGENTIME_CONFIG" \
  "$IMAGENTIME_WANDB_PROJECT" \
  "scripts_aireadi/train_imagentime_glucose_npy.sh")

DIFFUSIONTS_JOB=$(submit_job \
  "glucose_diffusionts" \
  "$DIFFUSIONTS_CONFIG" \
  "$DIFFUSIONTS_WANDB_PROJECT" \
  "scripts_aireadi/train_diffusionts_glucose_npy.sh")

echo "Submitted glucose ImagenTime job:   $IMAGENTIME_JOB"
echo "Submitted glucose DiffusionTS job:  $DIFFUSIONTS_JOB"
echo "Logs: $ROOT_DIR/logs/slurm"
