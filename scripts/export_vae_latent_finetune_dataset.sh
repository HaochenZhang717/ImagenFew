CUDA_VISIBLE_DEVICES=0 python export_vae_latents_finetune_dataset.py \
  --vae-config ../configs/pretrain/vae_pretrain.yaml \
  --vae-ckpt ../logs/vae_pretrain/52131940-4465-42cb-85b4-93bd7f5ee944/MultiScaleVAE.pt \
  --dataset-config-dir configs/DiffusionTS \
  --output-dir ../logs/vae_latents/finetune_dataset \
  --batch-size 256