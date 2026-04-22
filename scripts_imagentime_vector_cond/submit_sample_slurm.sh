#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/playpen-shared/haochenz/ImagenFew}"
SCRIPT_DIR="$ROOT_DIR/scripts_imagentime_vector_cond"
cd "$SCRIPT_DIR"

CONDA_ENV_VALUE="${CONDA_ENV:-vlm}"
SLURM_TIME_VALUE="${SLURM_TIME:-2-00:00:00}"
PYTHON_BIN_VALUE="${PYTHON_BIN:-}"

if [[ -z "$PYTHON_BIN_VALUE" ]]; then
  if [[ -x "/playpen/haochenz/miniconda3/envs/$CONDA_ENV_VALUE/bin/python" ]]; then
    PYTHON_BIN_VALUE="/playpen/haochenz/miniconda3/envs/$CONDA_ENV_VALUE/bin/python"
  elif [[ -x "/playpen-shared/haochenz/miniconda3/envs/$CONDA_ENV_VALUE/bin/python" ]]; then
    PYTHON_BIN_VALUE="/playpen-shared/haochenz/miniconda3/envs/$CONDA_ENV_VALUE/bin/python"
  elif [[ -x "$HOME/miniconda3/envs/$CONDA_ENV_VALUE/bin/python" ]]; then
    PYTHON_BIN_VALUE="$HOME/miniconda3/envs/$CONDA_ENV_VALUE/bin/python"
  elif [[ -x "$HOME/anaconda3/envs/$CONDA_ENV_VALUE/bin/python" ]]; then
    PYTHON_BIN_VALUE="$HOME/anaconda3/envs/$CONDA_ENV_VALUE/bin/python"
  else
    echo "Could not find python for conda env '$CONDA_ENV_VALUE'." >&2
    exit 1
  fi
fi

ETTM1_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" PYTHON_BIN="$PYTHON_BIN_VALUE" DATASET="ETTm1" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_ETTm1" "$SCRIPT_DIR/sample_slurm.sh")
SYN_U_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" PYTHON_BIN="$PYTHON_BIN_VALUE" DATASET="synthetic_u" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_synu" "$SCRIPT_DIR/sample_slurm.sh")
SYN_M_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" PYTHON_BIN="$PYTHON_BIN_VALUE" DATASET="synthetic_m" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_synm" "$SCRIPT_DIR/sample_slurm.sh")
ISTANBUL_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" PYTHON_BIN="$PYTHON_BIN_VALUE" DATASET="istanbul_traffic" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_istanbul" "$SCRIPT_DIR/sample_slurm.sh")

echo "Submitted ETTm1 job:            $ETTM1_JOB"
echo "Submitted synthetic_u job:      $SYN_U_JOB"
echo "Submitted synthetic_m job:      $SYN_M_JOB"
echo "Submitted istanbul_traffic job: $ISTANBUL_JOB"
