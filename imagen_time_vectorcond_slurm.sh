#!/usr/bin/env bash
#SBATCH --job-name=imtvcond
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT_DIR"
mkdir -p "$ROOT_DIR/logs/slurm"

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-vlm}" ]]; then
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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

WANDB_PROJECT="${WANDB_PROJECT:-ImagenTimeVectorCond-VerbalTS}"
SUBSET_P="${SUBSET_P:-1.0}"
USE_WANDB="${USE_WANDB:-1}"
IMTVCOND_LAUNCH_MODE="${IMTVCOND_LAUNCH_MODE:-submit}"

DEFAULT_CONFIGS=(
  "./configs/ImagenTimeVectorCond/VerbalTS_ETTm1_qwen3.yaml"
  "./configs/ImagenTimeVectorCond/VerbalTS_synthetic_u_qwen3.yaml"
  "./configs/ImagenTimeVectorCond/VerbalTS_synthetic_m_qwen3.yaml"
  "./configs/ImagenTimeVectorCond/VerbalTS_istanbul_traffic_qwen3.yaml"
)

if [[ -n "${CONFIG:-}" ]]; then
  CONFIGS=("$CONFIG")
elif [[ -n "${CONFIGS:-}" ]]; then
  read -r -a CONFIGS <<<"$CONFIGS"
else
  CONFIGS=("${DEFAULT_CONFIGS[@]}")
fi

echo "Running ImagenTimeVectorCond jobs on host $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "SUBSET_P=$SUBSET_P"
echo "USE_WANDB=$USE_WANDB"
echo "Configs:"
printf '  %s\n' "${CONFIGS[@]}"

if [[ "$IMTVCOND_LAUNCH_MODE" != "run" ]]; then
  echo "Submitting one Slurm job per config."
  for config in "${CONFIGS[@]}"; do
    config_base="$(basename "$config" .yaml)"
    job_name="imtvc_${config_base#VerbalTS_}"
    job_id="$(
      CONDA_ENV="${CONDA_ENV:-}" \
      WANDB_PROJECT="$WANDB_PROJECT" \
      SUBSET_P="$SUBSET_P" \
      USE_WANDB="$USE_WANDB" \
      CONFIG="$config" \
      IMTVCOND_LAUNCH_MODE="run" \
      sbatch --parsable --job-name="$job_name" "$ROOT_DIR/imagen_time_vectorcond_slurm.sh" "$@"
    )"
    echo "Submitted $config as job $job_id ($job_name)"
  done
  exit 0
fi

if [[ "${#CONFIGS[@]}" -ne 1 ]]; then
  echo "Run mode expects exactly one config, got ${#CONFIGS[@]}." >&2
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  CMD=(
    python run_diffts_imagentime.py
    --subset_p "$SUBSET_P"
    --config "$config"
  )

  if [[ "$USE_WANDB" == "1" ]]; then
    CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
  fi

  if [[ $# -gt 0 ]]; then
    CMD+=("$@")
  fi

  printf 'Running command:\n  %q' "${CMD[0]}"
  for arg in "${CMD[@]:1}"; do
    printf ' %q' "$arg"
  done
  printf '\n'

  "${CMD[@]}"
done
