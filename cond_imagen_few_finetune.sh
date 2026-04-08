#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs/conditional_imagen_few"
mkdir -p "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0 python "$ROOT_DIR/run.py" \
  --subset_p 1.0 \
  --wandb \
  --wandb_project CondImagenFewFinetune \
  --config "$ROOT_DIR/configs/conditional_imagen_few/ETTh2.yaml" \
  > "$LOG_DIR/ETTh2_gpu0.log" 2>&1 &
PID_ETTH2=$!

CUDA_VISIBLE_DEVICES=2 python "$ROOT_DIR/run.py" \
  --subset_p 1.0 \
  --wandb \
  --wandb_project CondImagenFewFinetune \
  --config "$ROOT_DIR/configs/conditional_imagen_few/mujoco.yaml" \
  > "$LOG_DIR/mujoco_gpu2.log" 2>&1 &
PID_MUJOCO=$!

CUDA_VISIBLE_DEVICES=4 python "$ROOT_DIR/run.py" \
  --subset_p 1.0 \
  --wandb \
  --wandb_project CondImagenFewFinetune \
  --config "$ROOT_DIR/configs/conditional_imagen_few/AirQuality.yaml" \
  > "$LOG_DIR/AirQuality_gpu4.log" 2>&1 &
PID_AIRQUALITY=$!

echo "Started ETTh2 on GPU 0 (pid: $PID_ETTH2)"
echo "Started mujoco on GPU 2 (pid: $PID_MUJOCO)"
echo "Started AirQuality on GPU 4 (pid: $PID_AIRQUALITY)"
echo "Logs:"
echo "  $LOG_DIR/ETTh2_gpu0.log"
echo "  $LOG_DIR/mujoco_gpu2.log"
echo "  $LOG_DIR/AirQuality_gpu4.log"

wait "$PID_ETTH2" "$PID_MUJOCO" "$PID_AIRQUALITY"








#CUDA_VISIBLE_DEVICES=0 python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project CondImagenFewFinetune \
#--config ./configs/conditional_imagen_few/ETTh2.yaml

#CUDA_VISIBLE_DEVICES=2 python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project CondImagenFewFinetune \
#--config ./configs/conditional_imagen_few/mujoco.yaml
#
#CUDA_VISIBLE_DEVICES=2 python run.py \
#--subset_p 1.0 \
#--wandb \
#--wandb_project CondImagenFewFinetune \
#--config ./configs/conditional_imagen_few/AirQuality.yaml

