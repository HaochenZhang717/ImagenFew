CUDA_VISIBLE_DEVICES=0 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project SelfConditionalGenerationFromScratch \
--config ./configs/self_cond_from_scratch/ETTh2.yaml


CUDA_VISIBLE_DEVICES=1 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project SelfConditionalGenerationFromScratch \
--config ./configs/self_cond_from_scratch/AirQuality.yaml


CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project SelfConditionalGenerationFromScratch \
--config ./configs/self_cond_from_scratch/mujoco.yaml


