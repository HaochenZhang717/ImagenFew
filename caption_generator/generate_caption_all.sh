#for d in ettm1 synthetic_m synthetic_u istanbul_traffic; do CONDA_ENV=vlm DATASET="$d" sbatch slurm_generate_caption.sh; done
for d in synthetic_m; do CONDA_ENV=vlm DATASET="$d" sbatch slurm_generate_caption.sh; done
