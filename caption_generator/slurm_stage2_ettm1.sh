#!/usr/bin/env bash
#SBATCH --job-name=caption_stage2_ettm1
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
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

CONFIG="${CONFIG:-configs/ettm1_stage2_diffusion_prior.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29688}"
STAGE1_CONFIG_PATH="${STAGE1_CONFIG_PATH:-./configs/ettm1_stage1_qwen25_3b.yaml}"
STAGE1_CKPT_PATH="${STAGE1_CKPT_PATH:-/playpen-shared/haochenz/ImagenFew/caption_generator/logs/caption_generator/ettm1_stage1_qwen25_3b/joint_caption_best.pt}"

echo "Running caption Stage 2 ETTm1 on host $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<managed by slurm>}"
echo "CONFIG=$CONFIG"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "MASTER_PORT=$MASTER_PORT"
echo "STAGE1_CONFIG_PATH=$STAGE1_CONFIG_PATH"
echo "STAGE1_CKPT_PATH=$STAGE1_CKPT_PATH"
echo "HF_HOME=$HF_HOME"

torchrun \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  "$WORK_DIR/train_stage2.py" \
  --config "$CONFIG" \
  --override \
  training.ddp=true \
  stage1.config_path="$STAGE1_CONFIG_PATH" \
  stage1.checkpoint_path="$STAGE1_CKPT_PATH" \
  "$@"
