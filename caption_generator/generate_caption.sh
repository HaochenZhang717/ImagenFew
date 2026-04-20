
export HF_HOME="${HF_HOME:-/playpen-shared/haochenz/hf_cache}"

#python generate_stage2.py \
#  --checkpoint ./logs/caption_generator/ettm1_stage2_diffusion_prior/stage2_best.pt \
#  --config ./configs/ettm1_stage2_diffusion_prior.yaml \
#  --use-ema \
#  --num-samples 8 \
#  --retrieve-train \
#  --retrieve-train-topk 5

python generate_stage2.py \
  --checkpoint ./logs/caption_generator/ettm1_stage2_diffusion_prior/stage2_best.pt \
  --config ./configs/ettm1_stage2_diffusion_prior.yaml \
  --use-ema \
  --output-npy ../data/VerbalTSDatasets/ETTm1/generated_text_caps.npy

python generate_stage2.py \
  --checkpoint ./logs/caption_generator/synthetic_m_stage2_diffusion_prior/stage2_best.pt \
  --config ./configs/synthetic_m_stage2_diffusion_prior.yaml \
  --use-ema \
  --output-npy ../data/VerbalTSDatasets/synthetic_m/generated_text_caps.npy

python generate_stage2.py \
  --checkpoint ./logs/caption_generator/synthetic_u_stage2_diffusion_prior/stage2_best.pt \
  --config ./configs/synthetic_u_stage2_diffusion_prior.yaml \
  --use-ema \
  --output-npy ../data/VerbalTSDatasets/synthetic_u/generated_text_caps.npy


python generate_stage2.py \
  --checkpoint ./logs/caption_generator/istanbul_stage2_diffusion_prior/stage2_best.pt \
  --config ./configs/istanbul_stage2_diffusion_prior.yaml \
  --use-ema \
  --output-npy ../data/VerbalTSDatasets/istanbul_traffic/generated_text_caps.npy
