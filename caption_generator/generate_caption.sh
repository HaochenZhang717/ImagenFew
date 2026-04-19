python generate_stage2.py \
  --checkpoint ./logs/caption_generator/ettm1_stage2_diffusion_prior/stage2_best.pt \
  --config ./configs/ettm1_stage2_diffusion_prior.yaml \
  --use-ema \
  --num-samples 8 \
  --retrieve-train \
  --retrieve-train-topk 5


#python generate_stage2.py \
#  --checkpoint /playpen-shared/haochenz/ImagenFew/caption_generator/logs/caption_generator/ettm1_stage2_diffusion_prior/stage2_best.pt \
#  --config caption_generator/configs/ettm1_stage2_diffusion_prior.yaml \
#  --use-ema