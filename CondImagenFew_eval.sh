CONDA_ENV=vlm \
MODEL_CKPT=./logs/CondImagenFewFinetune/ETTh2/19c5b45b-e429-4f16-96dc-180ad93dd2df/ImagenFewCrossAttention.pt \
PRIOR_CKPT=./logs/diffusion_prior/time_shift_sweep/ETTh2/tds_1p0/diffusion_prior_latest.pt \
CONFIG= ./configs/conditional_imagen_few/ETTh2.yaml
bash evaluate_imagenfew_cross_attention_prior_local.sh
