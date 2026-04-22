#!/usr/bin/env bash
#SBATCH --job-name=caption_stage1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  WORK_DIR="$SLURM_SUBMIT_DIR"
else
  WORK_DIR="$SCRIPT_DIR"
fi
cd "$WORK_DIR"
mkdir -p /playpen-shared/haochenz/logs/slurm

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

export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DATASET="${DATASET:-ettm1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

case "${DATASET}" in
  ettm1)
    CONFIG="${CONFIG:-configs/ettm1_stage1_qwen25_3b.yaml}"
    ;;
  istanbul|istanbul_traffic)
    DATASET="istanbul"
    CONFIG="${CONFIG:-configs/istanbul_stage1_qwen25_3b.yaml}"
    ;;
  synthetic_u)
    CONFIG="${CONFIG:-configs/synthetic_u_stage1_qwen25_3b.yaml}"
    ;;
  synthetic_m)
    CONFIG="${CONFIG:-configs/synthetic_m_stage1_qwen25_3b.yaml}"
    ;;
  blindways|BlindWays)
    DATASET="blindways"
    CONFIG="${CONFIG:-configs/blindways_stage1_qwen25_3b.yaml}"
    ;;
  weather|Weather)
    DATASET="weather"
    CONFIG="${CONFIG:-configs/weather_stage1_qwen25_3b.yaml}"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

echo "Running caption Stage 1 for dataset ${DATASET} on host $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<managed by slurm>}"
echo "CONFIG=$CONFIG"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "HF_HOME=$HF_HOME"

CMD=(
  torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  "$WORK_DIR/train_stage1.py"
  --config "$CONFIG"
  --override
  training.ddp=true
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'FINAL_CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
