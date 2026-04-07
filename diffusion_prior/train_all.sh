#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs/diffusion_prior"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=2 bash "$ROOT_DIR/diffusion_prior/train_small.sh" \
  > "$LOG_DIR/train_small_gpu2.log" 2>&1 &
PID_SMALL=$!

CUDA_VISIBLE_DEVICES=3 bash "$ROOT_DIR/diffusion_prior/train_base.sh" \
  > "$LOG_DIR/train_base_gpu3.log" 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES=4 bash "$ROOT_DIR/diffusion_prior/train_large.sh" \
  > "$LOG_DIR/train_large_gpu4.log" 2>&1 &
PID_LARGE=$!

echo "Started small on GPU 2  (pid: $PID_SMALL)"
echo "Started base  on GPU 3  (pid: $PID_BASE)"
echo "Started large on GPU 4  (pid: $PID_LARGE)"
echo "Logs:"
echo "  $LOG_DIR/train_small_gpu2.log"
echo "  $LOG_DIR/train_base_gpu3.log"
echo "  $LOG_DIR/train_large_gpu4.log"

wait "$PID_SMALL" "$PID_BASE" "$PID_LARGE"
