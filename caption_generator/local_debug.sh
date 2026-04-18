export HF_HOME=/playpen-shared/haochenz/hf_cache

CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
  --config configs/ettm1_stage1_qwen25_3b.yaml