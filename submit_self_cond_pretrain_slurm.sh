#!/usr/bin/env bash

CONDA_ENV="${CONDA_ENV:-vlm}" \
sbatch --time="${SLURM_TIME:-2-00:00:00}" self_cond_pretrain_slurm.sh
