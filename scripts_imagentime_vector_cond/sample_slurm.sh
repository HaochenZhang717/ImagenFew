#!/usr/bin/env bash
#SBATCH --job-name=imtvcsample
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

ROOT_DIR="${ROOT_DIR:-/playpen-shared/haochenz/ImagenFew}"
cd "$ROOT_DIR"
mkdir -p /playpen-shared/haochenz/logs/slurm

source ~/.zshrc >/dev/null 2>&1 || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  CONDA_BIN=""
#  if [[ -x "/playpen/haochenz/miniconda3/bin/conda" ]]; then
#    CONDA_BIN="/playpen/haochenz/miniconda3/bin/conda"
  if [[ -x "/playpen-shared/haochenz/miniconda3/bin/conda" ]]; then
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

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_ENV:-}" ]]; then
    if [[ -x "/playpen/haochenz/miniconda3/envs/$CONDA_ENV/bin/python" ]]; then
      PYTHON_BIN="/playpen/haochenz/miniconda3/envs/$CONDA_ENV/bin/python"
    elif [[ -x "/playpen-shared/haochenz/miniconda3/envs/$CONDA_ENV/bin/python" ]]; then
      PYTHON_BIN="/playpen-shared/haochenz/miniconda3/envs/$CONDA_ENV/bin/python"
    elif [[ -x "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" ]]; then
      PYTHON_BIN="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
    elif [[ -x "$HOME/anaconda3/envs/$CONDA_ENV/bin/python" ]]; then
      PYTHON_BIN="$HOME/anaconda3/envs/$CONDA_ENV/bin/python"
    else
      echo "Could not find python for conda env '$CONDA_ENV'." >&2
      exit 1
    fi
#  elif [[ -x "/playpen/haochenz/miniconda3/bin/python" ]]; then
#    PYTHON_BIN="/playpen/haochenz/miniconda3/bin/python"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/bin/python" ]]; then
    PYTHON_BIN="/playpen-shared/haochenz/miniconda3/bin/python"
  elif [[ -x "$HOME/miniconda3/bin/python" ]]; then
    PYTHON_BIN="$HOME/miniconda3/bin/python"
  elif [[ -x "$HOME/anaconda3/bin/python" ]]; then
    PYTHON_BIN="$HOME/anaconda3/bin/python"
  else
    echo "Could not find a usable absolute python binary." >&2
    exit 1
  fi
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

LOG_ROOT="${LOG_ROOT:-/playpen-shared/haochenz/ImagenFew/logs/ImagenTimeVectorCond}"
DATA_ROOT="${DATA_ROOT:-/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/playpen-shared/haochenz/ImagenFew/scripts_imagentime_vector_cond}"
METRIC_ITERATION="${METRIC_ITERATION:-10}"
REFERENCE_SPLIT="${REFERENCE_SPLIT:-train}"
SEED="${SEED:-0}"
EVAL_METRICS="${EVAL_METRICS:-disc vaeFID}"
FID_VAE_CKPT_ROOT="${FID_VAE_CKPT_ROOT:-/playpen-shared/haochenz/ImagenFew/fid_vae_ckpts}"
TS2VEC_ROOT="${TS2VEC_ROOT:-$OUTPUT_ROOT/TS2VEC}"
RUN_SUFFIX="${RUN_SUFFIX:-gs1p0_ld0p0_sp1p0}"

DATASET="${DATASET:-${1:-}}"
if [[ -z "$DATASET" ]]; then
  echo "Usage: DATASET=<dataset> sbatch /playpen-shared/haochenz/ImagenFew/scripts_imagentime_vector_cond/sample_slurm.sh" >&2
  exit 1
fi

declare -A CONFIG_MAP=(
  ["ETTm1"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_ETTm1_qwen3.yaml"
  ["synthetic_u"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_synthetic_u_qwen3.yaml"
  ["synthetic_m"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_synthetic_m_qwen3.yaml"
  ["istanbul_traffic"]="$ROOT_DIR/configs/ImagenTimeVectorCond/VerbalTS_istanbul_traffic_qwen3.yaml"
)

declare -A RUN_DIR_MAP=(
  ["ETTm1"]="$LOG_ROOT/VerbalTS_ETTm1_qwen3/ETTm1_qwen3_${RUN_SUFFIX}"
  ["synthetic_u"]="$LOG_ROOT/VerbalTS_synthetic_u_qwen3/synthetic_u_qwen3_${RUN_SUFFIX}"
  ["synthetic_m"]="$LOG_ROOT/VerbalTS_synthetic_m_qwen3/synthetic_m_qwen3_${RUN_SUFFIX}"
  ["istanbul_traffic"]="$LOG_ROOT/VerbalTS_istanbul_traffic_qwen3/istanbul_traffic_qwen3_${RUN_SUFFIX}"
)

if [[ -z "${CONFIG_MAP[$DATASET]:-}" ]]; then
  echo "Unsupported dataset: $DATASET" >&2
  exit 1
fi

config_path="${CONFIG_MAP[$DATASET]}"
run_dir="${RUN_DIR_MAP[$DATASET]}"
ckpt_dir="$run_dir/checkpoints"
embeds_path="$DATA_ROOT/$DATASET/generated_embeds_qwen3_4b.pt"
samples_dir="$run_dir/eval_samples_generated_qwen3"
jsonl_path="$OUTPUT_ROOT/${DATASET}_generated_qwen3_metrics.jsonl"
ts2vec_dir="$TS2VEC_ROOT/$DATASET"

mkdir -p "$OUTPUT_ROOT" "$TS2VEC_ROOT"

if [[ ! -f "$config_path" ]]; then
  echo "Missing config: $config_path" >&2
  exit 1
fi
if [[ ! -d "$ckpt_dir" ]]; then
  echo "Missing checkpoint dir: $ckpt_dir" >&2
  exit 1
fi
if [[ ! -f "$embeds_path" ]]; then
  echo "Missing embeddings: $embeds_path" >&2
  exit 1
fi

mkdir -p "$samples_dir" "$ts2vec_dir"
: > "$jsonl_path"

echo "=== Dataset: $DATASET ==="
echo "Python: $PYTHON_BIN"
echo "Config: $config_path"
echo "Checkpoint dir: $ckpt_dir"
echo "Embeddings: $embeds_path"
echo "Metrics jsonl: $jsonl_path"

mapfile -t ckpts < <(find "$ckpt_dir" -maxdepth 1 -type f -name 'epoch_*.pt' | sort -V)
if [[ "${#ckpts[@]}" -eq 0 ]]; then
  echo "No checkpoints found under $ckpt_dir" >&2
  exit 1
fi

for ckpt_path in "${ckpts[@]}"; do
  ckpt_tag="$(basename "$ckpt_path" .pt)"
  output_path="$samples_dir/${DATASET}_generated_embeds_qwen3_4b_${ckpt_tag}.pt"

  echo "--- Sampling $DATASET / $ckpt_tag ---"
  "$PYTHON_BIN" "$ROOT_DIR/scripts_imagentime_vector_cond/sample_imagen_time_vectorcond.py" \
    --config "$config_path" \
    --dataset "$DATASET" \
    --model-ckpt "$ckpt_path" \
    --embeds-path "$embeds_path" \
    --reference-split "$REFERENCE_SPLIT" \
    --seed "$SEED" \
    --metric-iteration "$METRIC_ITERATION" \
    --fid-vae-ckpt-root "$FID_VAE_CKPT_ROOT" \
    --ts2vec-dir "$ts2vec_dir" \
    --output "$output_path" \
    --output-jsonl "$jsonl_path" \
    --eval-metrics $EVAL_METRICS
done
