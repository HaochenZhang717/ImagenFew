#sbatch --export=ALL,DATASET=ettm1 slurm_stage2.sh
CONDA_ENV=vlm sbatch --export=ALL,DATASET=istanbul slurm_stage2.sh
CONDA_ENV=vlm sbatch --export=ALL,DATASET=synthetic_u slurm_stage2.sh
CONDA_ENV=vlm sbatch --export=ALL,DATASET=synthetic_m slurm_stage2.sh
#sbatch --export=ALL,DATASET=blindways slurm_stage2.sh
CONDA_ENV=vlm sbatch --export=ALL,DATASET=weather slurm_stage2.sh