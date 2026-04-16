#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-./configs/DiffusionTS/AIREADIGlucose.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-DiffusionTS}"
SUBSET_P="${SUBSET_P:-1.0}"
USE_WANDB="${USE_WANDB:-1}"
GPU="${GPU:-0}"

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

CMD=(
  python run.py
  --config "$CONFIG"
  --subset_p "$SUBSET_P"
)

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--wandb --wandb_project "$WANDB_PROJECT")
fi

CMD+=("$@")

printf 'Running command:\n  %q' "${CMD[0]}"
for arg in "${CMD[@]:1}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${CMD[@]}"
