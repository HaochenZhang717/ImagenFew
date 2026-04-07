#!/bin/bash
# Quick VAE reconstruction visualization.
#
# Edit CKPT / DATASET below, then run:   bash visualize_vae.sh

CONFIG="./configs/pretrain/vae_pretrain.yaml"
CKPT="/work/vb21/haochen/neurips_2026/ImagenFew/logs/vae_pretrain/52131940-4465-42cb-85b4-93bd7f5ee944/MultiScaleVAE"
DATASET="ETTh2"                            # <-- one of the names in the YAML 'datasets' list
SAVE_DIR="./vae_vis"

mkdir -p "$SAVE_DIR"

CUDA_VISIBLE_DEVICES=0 python visualize_vae.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --dataset "$DATASET" \
    --n_samples 16 \
    --sample_idx 0 \
    --vars 0 1 2 3 \
    --save "$SAVE_DIR/${DATASET}_recon.png"
