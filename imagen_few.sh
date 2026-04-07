CUDA_VISIBLE_DEVICES=1 python run.py \
--subset_p 1.0 \
--wandb \
--wandb_project ImagenFew \
--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
--config ./configs/finetune/AirQuality.yaml


#CUDA_VISIBLE_DEVICES=1 python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/ETTh2.yaml
#
#
#CUDA_VISIBLE_DEVICES=1 python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project ImagenFew \
#--model_ckpt ./ImagenFew_ckpts/ImagenFew_24.ckpt \
#--config ./configs/finetune/Mujoco.yaml



