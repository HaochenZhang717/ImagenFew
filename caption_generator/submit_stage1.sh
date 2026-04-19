#sbatch --export=ALL,DATASET=ettm1 slurm_stage1.sh
sbatch --export=ALL,DATASET=istanbul slurm_stage1.sh
sbatch --export=ALL,DATASET=synthetic_u slurm_stage1.sh
#sbatch --export=ALL,DATASET=synthetic_m slurm_stage1.sh
#sbatch --export=ALL,DATASET=blindways slurm_stage1.sh
#sbatch --export=ALL,DATASET=weather slurm_stage1.sh
