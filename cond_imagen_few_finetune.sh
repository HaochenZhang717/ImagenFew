CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project CondImagenFewFinetune \
--config ./configs/conditional_imagen_few/ETTh2.yaml

CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project CondImagenFewFinetune \
--config ./configs/conditional_imagen_few/mujoco.yaml

CUDA_VISIBLE_DEVICES=2 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project CondImagenFewFinetune \
--config ./configs/conditional_imagen_few/AirQuality.yaml

