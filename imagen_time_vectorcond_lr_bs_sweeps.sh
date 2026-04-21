#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV_VALUE="${CONDA_ENV:-vlm}"
WANDB_PROJECT_VALUE="${WANDB_PROJECT:-ImagenTimeVectorCond-VerbalTS}"
USE_WANDB_VALUE="${USE_WANDB:-1}"
SUBSET_P_VALUE="${SUBSET_P:-1.0}"
GUIDANCE_SCALE_VALUE="${GUIDANCE_SCALE:-1.0}"
LABEL_DROPOUT_VALUE="${LABEL_DROPOUT:-0.0}"

# Coarse sweep grids.
LR_GRID="${LR_GRID:-3e-5 1e-4 3e-4}"
BS_GRID="${LARGE_BS_GRID:-1024 2048 4096}"
#LARGE_BS_GRID="${LARGE_BS_GRID:-1024 2048 4096}"
#MEDIUM_BS_GRID="${MEDIUM_BS_GRID:-512 1024 2048}"
#SMALL_BS_GRID="${SMALL_BS_GRID:-64 128 256}"

submit_sweep() {
  local config_path="$1"
  local batch_grid="$2"
  local suffix="$3"

  CONDA_ENV="$CONDA_ENV_VALUE" \
  WANDB_PROJECT="$WANDB_PROJECT_VALUE" \
  USE_WANDB="$USE_WANDB_VALUE" \
  SUBSET_P="$SUBSET_P_VALUE" \
  GUIDANCE_SCALE="$GUIDANCE_SCALE_VALUE" \
  LABEL_DROPOUT="$LABEL_DROPOUT_VALUE" \
  LR_VALUES="$LR_GRID" \
  BATCH_SIZE_VALUES="$batch_grid" \
  RUN_SUFFIX="$suffix" \
  CONFIG="$config_path" \
  sbatch "$ROOT_DIR/imagen_time_vectorcond_slurm.sh"
}

submit_sweep "./configs/ImagenTimeVectorCond/VerbalTS_ETTm1_qwen3.yaml" "$BS_GRID" "coarse_lrbs"
submit_sweep "./configs/ImagenTimeVectorCond/VerbalTS_synthetic_m_qwen3.yaml" "$BS_GRID" "coarse_lrbs"
submit_sweep "./configs/ImagenTimeVectorCond/VerbalTS_synthetic_u_qwen3.yaml" "$BS_GRID" "coarse_lrbs"
#submit_sweep "./configs/ImagenTimeVectorCond/VerbalTS_Weather_qwen3.yaml" "$MEDIUM_BS_GRID" "coarse_lrbs"
submit_sweep "./configs/ImagenTimeVectorCond/VerbalTS_istanbul_traffic_qwen3.yaml" "$BS_GRID" "coarse_lrbs"
#submit_sweep "./configs/ImagenTimeVectorCond/VerbalTS_BlindWays_qwen3.yaml" "$SMALL_BS_GRID" "coarse_lrbs"
