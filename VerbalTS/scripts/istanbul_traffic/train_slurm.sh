#!/usr/bin/env bash
#SBATCH --job-name=verbalts_istanbul
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/playpen-shared/haochenz/logs/slurm/%x_%j.out
#SBATCH --error=/playpen-shared/haochenz/logs/slurm/%x_%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  WORK_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  WORK_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

if [[ -d "$WORK_DIR/VerbalTS" ]]; then
  PROJECT_DIR="$WORK_DIR/VerbalTS"
elif [[ "$(basename "$WORK_DIR")" == "VerbalTS" ]]; then
  PROJECT_DIR="$WORK_DIR"
else
  echo "Could not infer VerbalTS project directory from WORK_DIR=$WORK_DIR" >&2
  exit 1
fi

cd "$PROJECT_DIR"
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
export SCHEDULER="${SCHEDULER:-MULTISTEP}"

echo "Running VerbalTS istanbul_traffic training on host $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<managed by slurm>}"
echo "CONDA_ENV=${CONDA_ENV:-<none>}"

CMD=(
  python run.py
  --cond_modal text
  --training_stage finetune
  --save_folder ./logs/istanbul_traffic/text2ts_msmdiffmv
  --model_diff_config_path configs/istanbul_traffic/diff/model_text2ts_dep.yaml
  --model_cond_config_path configs/istanbul_traffic/cond/text_msmdiffmv.yaml
  --train_config_path configs/istanbul_traffic/train.yaml
  --evaluate_config_path configs/istanbul_traffic/evaluate.yaml
  --data_folder ../data/VerbalTSDatasets/istanbul_traffic
  --clip_folder ""
  --multipatch_num 3
  --L_patch_len 3
  --base_patch 4
  --epochs 700
  --batch_size 512
  --clip_cache_path ""
)

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'FINAL_CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
