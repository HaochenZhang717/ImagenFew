#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$ROOT_DIR/diffusion_prior/train_diffusion_prior.py" \
  --config "$ROOT_DIR/diffusion_prior/configs/dit1d_base.yaml" \
  "$@"
