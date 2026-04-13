#CUDA_VISIBLE_DEVICES=4 python run_no_sample.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project MultiScaleVAE \
#--config ./configs/pretrain/vae_pretrain.yaml



CUDA_VISIBLE_DEVICES=4 python run_no_sample.py \
--subset_p 1.0 \
--wandb \
--wandb_project MultiScaleVAE \
--config ./configs/pretrain/vae_pretrain_one_token.yaml