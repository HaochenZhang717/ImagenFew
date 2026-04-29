HF_HOME_DEFAULT="$HOME/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_ROOT="${DATASET_ROOT:-data/VerbalTSDatasets/istanbul_traffic}"
DATASET_NAME="${DATASET_NAME:-istanbul_traffic}"
SPLITS=(${SPLITS:-train valid test})
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_PARTS="${NUM_PARTS:-1}"
PART_ID="${PART_ID:-0}"

for split in "${SPLITS[@]}"; do
  echo "Running Qwen3-VL caption paraphrases for ${DATASET_NAME} split=${split}"
  python scripts/paraphrase_captions_qwen3vl.py \
    --input-npy "/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/${DATASET_ROOT}/${split}_my_text_caps.npy" \
    --output-npy "/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/${DATASET_ROOT}/${split}_my_text_caps_paraphrased.npy" \
    --save-dir "/playpen-shared/haochenz/ImagenFew/data/VerbalTSDatasets/${DATASET_NAME}" \
    --part_id "$PART_ID" \
    --num_parts "$NUM_PARTS" \
    --batch_size "$BATCH_SIZE" \
    --fallback-original-on-bad-format
done
