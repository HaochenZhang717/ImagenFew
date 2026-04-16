#!/usr/bin/env bash
#SBATCH --job-name=difts_aireadi
#SBATCH --partition=a6000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BIN=""
  if [[ -x "/playpen/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen/haochenz/miniconda3/bin/conda"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/bin/conda" ]]; then
    CONDA_BIN="/playpen-shared/haochenz/miniconda3/bin/conda"
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_BIN="$HOME/anaconda3/bin/conda"
  else
    echo "Could not find a usable conda binary." >&2
    exit 1
  fi
  eval "$("$CONDA_BIN" shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

CONFIG="${CONFIG:-./configs/DiffusionTS/AIREADIGlucose.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-DiffusionTS}"
SUBSET_P="${SUBSET_P:-1.0}"
USE_WANDB="${USE_WANDB:-1}"

CMD=(
  python run.py
  --config "$CONFIG"
  --subset_p "$SUBSET_P"
)

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
fi

CMD+=("$@")

printf 'Running command:\n  %q' "${CMD[0]}"
for arg in "${CMD[@]:1}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${CMD[@]}"
