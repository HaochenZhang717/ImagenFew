export HF_HOME=/playpen-shared/haochenz/hf_cache


OUT_DIR="/playpen-shared/haochenz/outputs_diffusion_prior_0306/stock"
mkdir ${OUT_DIR}
CUDA_VISIBLE_DEVICES=3 python sample_one_image_per_channel.py \
  --config "/playpen-shared/haochenz/ckpts_diffusion_prior_0306/stock/DiTDH-S/config.yaml" \
  --seed 42 \
  --num_samples 8000 \
  --num_channels 6  \
  --ckpt "/playpen-shared/haochenz/ckpts_diffusion_prior_0306/stock/DiTDH-S/checkpoints/ep-last.pt" \
  --output_jsonl "${OUT_DIR}/DiTDH-S-samples.jsonl" \
  --output_array "${OUT_DIR}/DiTDH-S-samples.npy" \
  --latent_size 1152 48 32

