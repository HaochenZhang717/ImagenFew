#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

#for script in \
#  "multi_scale_vae_finetune_airquality_slurm.sh" \
#  "multi_scale_vae_finetune_ecg200_slurm.sh" \
#  "multi_scale_vae_finetune_etth2_slurm.sh" \
#  "multi_scale_vae_finetune_ettm1_slurm.sh" \
#  "multi_scale_vae_finetune_ettm2_slurm.sh" \
#  "multi_scale_vae_finetune_ili_slurm.sh" \
#  "multi_scale_vae_finetune_mujoco_slurm.sh" \
#  "multi_scale_vae_finetune_saugeenriverflow_slurm.sh" \
#  "multi_scale_vae_finetune_selfregulationscp1_slurm.sh" \
#  "multi_scale_vae_finetune_sine_slurm.sh" \
#  "multi_scale_vae_finetune_starlightcurves_slurm.sh" \
#  "multi_scale_vae_finetune_weather_slurm.sh"
#do
#  echo "Submitting $script"
#  sbatch "$script"
#done



for script in \
  "multi_scale_vae_finetune_airquality_slurm.sh" \
  "multi_scale_vae_finetune_etth2_slurm.sh" \
  "multi_scale_vae_finetune_mujoco_slurm.sh"
do
  echo "Submitting $script"
  sbatch "$script"
done


