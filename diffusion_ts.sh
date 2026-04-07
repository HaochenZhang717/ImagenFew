CUDA_VISIBLE_DEVICES=0 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project Diffusion-TS \
--config ./configs/DiffusionTS/AirQuality.yaml


CUDA_VISIBLE_DEVICES=0 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project Diffusion-TS \
--config ./configs/DiffusionTS/ETTh2.yaml


CUDA_VISIBLE_DEVICES=0 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project Diffusion-TS \
--config ./configs/DiffusionTS/Mujoco.yaml



