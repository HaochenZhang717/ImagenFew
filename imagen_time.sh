CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenTime \
--config ./configs/ImagenTime/AirQuality.yaml


CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenTime \
--config ./configs/ImagenTime/ETTh2.yaml


CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenTime \
--config ./configs/ImagenTime/Mujoco.yaml



