export HF_HOME=/playpen-shared/haochenz/hf_cache
CONFIG="DiTDH-S.yaml"
CUDA_VISIBLE_DEVICES=1 python sample.py \
  --config ${CONFIG} \
  --seed 42 \
  --num_samples 24000 \
  --ckpt "/playpen-shared/haochenz/ckpts_diffusion_prior/DiTDH-S/checkpoints/ep-last.pt" \
  --output_jsonl "/playpen-shared/haochenz/diffusion_prior_results/DiTDH-S-samples.jsonl" \
  --output_array "/playpen-shared/haochenz/diffusion_prior_results/DiTDH-S-samples.npy"


