export HF_HOME=/playpen/haochenz/hf_cache
CONFIG="DiTDH-XL.yaml"
CUDA_VISIBLE_DEVICES=2 python sample.py \
  --config ${CONFIG} \
  --seed 42 \
  --num_samples 24000 \
  --ckpt "/playpen-shared/haochenz/ckpts_diffusion_prior/DiTDH-XL/checkpoints/ep-last.pt" \
  --output_jsonl "/playpen/haochenz/diffusion_prior_results/DiTDH-XL-samples.jsonl" \
  --output_array "/playpen/haochenz/diffusion_prior_results/DiTDH-XL-samples.npy"


