#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_DIR="${DATASET_DIR:-../data/VerbalTSDatasets/Weather}"
TRAIN_PATH="${TRAIN_PATH:-$DATASET_DIR/train_ts.npy}"
VAL_PATH="${VAL_PATH:-$DATASET_DIR/valid_ts.npy}"
SAVE_DIR="${SAVE_DIR:-./vae_ckpts/$(basename "$DATASET_DIR")}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-3}"
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"
NUM_LAYERS="${NUM_LAYERS:-2}"
NUM_HEADS="${NUM_HEADS:-8}"
LATENT_DIM="${LATENT_DIM:-128}"
BETA="${BETA:-0.001}"
SCALE="${SCALE:-1}"

CMD=(
  python train_fid_vae.py
  --train_path "$TRAIN_PATH"
  --val_path "$VAL_PATH"
  --save_dir "$SAVE_DIR"
  --device "$DEVICE"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --hidden_size "$HIDDEN_SIZE"
  --num_layers "$NUM_LAYERS"
  --num_heads "$NUM_HEADS"
  --latent_dim "$LATENT_DIM"
  --beta "$BETA"
)

if [[ "$SCALE" == "1" ]]; then
  CMD+=(--scale)
fi

if [[ $# -gt 0 ]]; then
  CMD+=("$@")
fi

printf 'Running command:\n  %q' "${CMD[0]}"
for arg in "${CMD[@]:1}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${CMD[@]}"
