#!/usr/bin/env bash
#SBATCH --job-name=caption_generate
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
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
USE_EMA="${USE_EMA:-1}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
OUTPUT_NPY="${OUTPUT_NPY:-}"

case "${DATASET}" in
  ettm1|ETTm1)
    DATASET_NAME="ETTm1"
    CHECKPOINT="${CHECKPOINT:-./logs/caption_generator/ettm1_stage2_diffusion_prior/stage2_best.pt}"
    CONFIG="${CONFIG:-./configs/ettm1_stage2_diffusion_prior.yaml}"
    DEFAULT_OUTPUT_NPY="../data/VerbalTSDatasets/ETTm1/generated_text_caps.npy"
    ;;
  synthetic_m)
    DATASET_NAME="synthetic_m"
    CHECKPOINT="${CHECKPOINT:-./logs/caption_generator/synthetic_m_stage2_diffusion_prior/stage2_best.pt}"
    CONFIG="${CONFIG:-./configs/synthetic_m_stage2_diffusion_prior.yaml}"
    DEFAULT_OUTPUT_NPY="../data/VerbalTSDatasets/synthetic_m/generated_text_caps.npy"
    ;;
  synthetic_u)
    DATASET_NAME="synthetic_u"
    CHECKPOINT="${CHECKPOINT:-./logs/caption_generator/synthetic_u_stage2_diffusion_prior/stage2_best.pt}"
    CONFIG="${CONFIG:-./configs/synthetic_u_stage2_diffusion_prior.yaml}"
    DEFAULT_OUTPUT_NPY="../data/VerbalTSDatasets/synthetic_u/generated_text_caps.npy"
    ;;
  istanbul|istanbul_traffic)
    DATASET_NAME="istanbul_traffic"
    CHECKPOINT="${CHECKPOINT:-./logs/caption_generator/istanbul_stage2_diffusion_prior/stage2_best.pt}"
    CONFIG="${CONFIG:-./configs/istanbul_stage2_diffusion_prior.yaml}"
    DEFAULT_OUTPUT_NPY="../data/VerbalTSDatasets/istanbul_traffic/generated_text_caps.npy"
    ;;
  blindways|BlindWays)
    DATASET_NAME="BlindWays"
    CHECKPOINT="${CHECKPOINT:-./logs/caption_generator/blindways_stage2_diffusion_prior/stage2_best.pt}"
    CONFIG="${CONFIG:-./configs/blindways_stage2_diffusion_prior.yaml}"
    DEFAULT_OUTPUT_NPY="../data/VerbalTSDatasets/BlindWays/generated_text_caps.npy"
    ;;
  weather|Weather)
    DATASET_NAME="Weather"
    CHECKPOINT="${CHECKPOINT:-./logs/caption_generator/weather_stage2_diffusion_prior/stage2_best.pt}"
    CONFIG="${CONFIG:-./configs/weather_stage2_diffusion_prior.yaml}"
    DEFAULT_OUTPUT_NPY="../data/VerbalTSDatasets/Weather/generated_text_caps.npy"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}" >&2
    exit 1
    ;;
esac

if [[ -z "$OUTPUT_NPY" ]]; then
  OUTPUT_NPY="$DEFAULT_OUTPUT_NPY"
fi

echo "Running caption generation for dataset ${DATASET_NAME} on host $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "WORK_DIR=$WORK_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<managed by slurm>}"
echo "CHECKPOINT=$CHECKPOINT"
echo "CONFIG=$CONFIG"
echo "OUTPUT_JSON=${OUTPUT_JSON:-<none>}"
echo "OUTPUT_NPY=$OUTPUT_NPY"
echo "USE_EMA=$USE_EMA"
echo "NUM_SAMPLES=${NUM_SAMPLES:-<match test split>}"

CMD=(
  python "$WORK_DIR/generate_stage2.py"
  --checkpoint "$CHECKPOINT"
  --config "$CONFIG"
  --output-npy "$OUTPUT_NPY"
)

if [[ "$USE_EMA" == "1" ]]; then
  CMD+=(--use-ema)
fi

if [[ -n "$NUM_SAMPLES" ]]; then
  CMD+=(--num-samples "$NUM_SAMPLES")
fi

if [[ -n "$OUTPUT_JSON" ]]; then
  CMD+=(--output "$OUTPUT_JSON")
fi

if [[ "$#" -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'FINAL_CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
