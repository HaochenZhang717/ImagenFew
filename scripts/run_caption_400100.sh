HF_HOME_DEFAULT="$HOME/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python scripts/run_caption_400x100.py --split test
