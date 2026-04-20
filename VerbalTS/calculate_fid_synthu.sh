#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=1


REAL_PATH="/playpen-shared/haochenz/sweep_text2ts/synth_u/orig_verbalts_orig_cap_precompute_embed-lr1e-3_bs512_multistep/0/samples.pt"
FAKE_PATH="/playpen-shared/haochenz/sweep_text2ts/synth_u/orig_verbalts_orig_cap_precompute_embed-lr1e-3_bs512_multistep/0/samples.pt"
python calculate_fid.py \
    --real_path ${REAL_PATH} \
    --fake_path ${FAKE_PATH} \
    --ckpt_path "../fid_vae_ckpts/vae_synth_u/best.pt" \
    --batch_size 128 \
    --hidden_size 128 \
    --num_layers 2 \
    --num_heads 8 \
    --latent_dim 64 \
    --save_path "./fid_results/verbal_ts.txt"