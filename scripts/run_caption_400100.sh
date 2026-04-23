HF_HOME_DEFAULT="$HOME/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_ROOT="${DATASET_ROOT:-data/VerbalTSDatasets/istanbul_traffic}"
DATASET_NAME="${DATASET_NAME:-istanbul_traffic}"
SPLITS=(${SPLITS:-train valid test})

for split in "${SPLITS[@]}"; do
  echo "Running 400x100 captions for ${DATASET_NAME} split=${split}"
  python scripts/run_caption_400x100.py \
    --dataset-root "$DATASET_ROOT" \
    --dataset_name "$DATASET_NAME" \
    --split "$split"
done
