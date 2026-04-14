#!/usr/bin/env bash

#CONDA_ENV="${CONDA_ENV:-vlm}" \
#sbatch --time="${SLURM_TIME:-2-00:00:00}" self_cond_finetune_etth2_slurm.sh

CONDA_ENV="${CONDA_ENV:-vlm}" \
sbatch --time="${SLURM_TIME:-2-00:00:00}" self_cond_finetune_airquality_slurm.sh

CONDA_ENV="${CONDA_ENV:-vlm}" \
sbatch --time="${SLURM_TIME:-2-00:00:00}" self_cond_finetune_mujoco_slurm.sh
