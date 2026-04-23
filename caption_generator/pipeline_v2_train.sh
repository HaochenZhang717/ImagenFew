HF_HOME_DEFAULT="$HOME/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"



#python prepare_pipeline_v2_dataset.py --config configs/synthetic_u_pipeline_v2_qwen3vl.yaml



CUDA_VISIBLE_DEVICES=3 python train_pipeline_v2.py --config configs/ettm1_pipeline_v2_qwen3vl.yaml