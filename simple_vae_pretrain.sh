CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python run_no_sample.py \
  --subset_p "${SUBSET_P:-1.0}" \
  --wandb \
  --wandb_project "${WANDB_PROJECT:-SimpleVAE}" \
  --config "${CONFIG:-./configs/pretrain/simple_vae_pretrain.yaml}"

