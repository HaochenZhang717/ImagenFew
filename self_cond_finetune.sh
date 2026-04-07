CUDA_VISIBLE_DEVICES=0 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project SelfConditionalGenerationFinetune \
--config ./configs/self_cond_finetune/ETTh2.yaml


CUDA_VISIBLE_DEVICES=1 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project SelfConditionalGenerationFinetune \
--config ./configs/self_cond_finetune/AirQuality.yaml


CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project SelfConditionalGenerationFinetune \
--config ./configs/self_cond_finetune/mujoco.yaml


