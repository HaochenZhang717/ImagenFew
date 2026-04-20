#!/bin/bash

# =========================
# CONFIG
# =========================
CAPS_PATH="../../data/VerbalTSDatasets/synthetic_u"
SAVE_ROOT="../../data/VerbalTSDatasets/synthetic_u"

SCRIPT_PATH="precompute_long_clip_embeds.py"
NPY_NAME="text_caps"

BATCH_SIZE=512
DEVICE="cuda"

mkdir -p $SAVE_ROOT

echo "Start precomputing LongClip embeddings..."

# =========================
# TRAIN
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/train_embeds_long_clip.pt \
    --npy_name ${NPY_NAME} \
    --split train \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# VAL
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/valid_embeds_long_clip.pt \
    --npy_name ${NPY_NAME} \
    --split valid \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# TEST
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/test_embeds_long_clip.pt \
    --npy_name ${NPY_NAME} \
    --split test \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/generated_embeds_long_clip.pt \
    --npy_name ${NPY_NAME} \
    --split generated \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

echo "Done!"