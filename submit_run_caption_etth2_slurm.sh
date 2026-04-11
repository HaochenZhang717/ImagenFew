#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-vlm}" \
sbatch --time="${SLURM_TIME:-2-00:00:00}" run_caption_etth2_part0_slurm.sh

CONDA_ENV="${CONDA_ENV:-vlm}" \
sbatch --time="${SLURM_TIME:-2-00:00:00}" run_caption_etth2_part1_slurm.sh

CONDA_ENV="${CONDA_ENV:-vlm}" \
sbatch --time="${SLURM_TIME:-2-00:00:00}" run_caption_etth2_part2_slurm.sh

CONDA_ENV="${CONDA_ENV:-vlm}" \
sbatch --time="${SLURM_TIME:-2-00:00:00}" run_caption_etth2_part3_slurm.sh
