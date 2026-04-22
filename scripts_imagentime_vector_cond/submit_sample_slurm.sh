#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV_VALUE="${CONDA_ENV:-vlm}"
SLURM_TIME_VALUE="${SLURM_TIME:-2-00:00:00}"

ETTM1_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" DATASET="ETTm1" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_ETTm1" "$SCRIPT_DIR/sample_slurm.sh")
SYN_U_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" DATASET="synthetic_u" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_synu" "$SCRIPT_DIR/sample_slurm.sh")
SYN_M_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" DATASET="synthetic_m" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_synm" "$SCRIPT_DIR/sample_slurm.sh")
ISTANBUL_JOB=$(CONDA_ENV="$CONDA_ENV_VALUE" DATASET="istanbul_traffic" sbatch --parsable --time="$SLURM_TIME_VALUE" --job-name="imtvcsample_istanbul" "$SCRIPT_DIR/sample_slurm.sh")

echo "Submitted ETTm1 job:            $ETTM1_JOB"
echo "Submitted synthetic_u job:      $SYN_U_JOB"
echo "Submitted synthetic_m job:      $SYN_M_JOB"
echo "Submitted istanbul_traffic job: $ISTANBUL_JOB"
