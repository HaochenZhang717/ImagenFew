CUDA_VISIBLE_DEVICES=2 python scripts/export_vae_latents.py \
  --config configs/pretrain/vae_pretrain.yaml \
  --ckpt ./logs/vae_pretrain/52131940-4465-42cb-85b4-93bd7f5ee944/MultiScaleVAE \
  --output ./logs/vae_latents/pretrain_latents.pt
