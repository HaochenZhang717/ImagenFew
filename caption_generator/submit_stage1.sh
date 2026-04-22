CONDA_ENV=vlm sbatch --export=ALL,DATASET=ettm1,NPROC_PER_NODE=4 slurm_stage1.sh
#CONDA_ENV=vlm sbatch --export=ALL,DATASET=istanbul,NPROC_PER_NODE=1 slurm_stage1.sh
#CONDA_ENV=vlm sbatch --export=ALL,DATASET=synthetic_u,NPROC_PER_NODE=1 slurm_stage1.sh
#CONDA_ENV=vlm sbatch --export=ALL,DATASET=synthetic_m,NPROC_PER_NODE=1 slurm_stage1.sh
#CONDA_ENV=vlm sbatch --export=ALL,DATASET=blindways,NPROC_PER_NODE=1 slurm_stage1.sh
#CONDA_ENV=vlm sbatch --export=ALL,DATASET=weather,NPROC_PER_NODE=1 slurm_stage1.sh
