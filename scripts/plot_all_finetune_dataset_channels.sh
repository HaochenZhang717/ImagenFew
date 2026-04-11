#!/usr/bin/env bash


OUTPUT_DIR="${OUTPUT_DIR:-./plots/self_cond_finetune_channels}"
SPLIT="${SPLIT:-train}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

CONFIGS=(
  "./configs/self_cond_finetune/AirQuality.yaml"
#  "./configs/self_cond_finetune/ETTh2.yaml"
#  "./configs/self_cond_finetune/ETTm1.yaml"
#  "./configs/self_cond_finetune/ETTm2.yaml"
#  "./configs/self_cond_finetune/ILI.yaml"
#  "./configs/self_cond_finetune/SaugeenRiverFlow.yaml"
#  "./configs/self_cond_finetune/SelfRegulationSCP1.yaml"
#  "./configs/self_cond_finetune/Sine.yaml"
#  "./configs/self_cond_finetune/StarLightCurves.yaml"
#  "./configs/self_cond_finetune/Weather.yaml"
#  "./configs/self_cond_finetune/mujoco.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo "Plotting channels for $config"
  cmd=(
    python scripts/plot_dataset_channels.py
    --config "$config"
    --split "$SPLIT"
    --output-dir "$OUTPUT_DIR"
  )

  cmd=(
    python scripts/plot_dataset_channels.py \
    --config "$config" \
    --split "$SPLIT" \
    --max-samples 100 \
    --num-workers 8
  )



  if [[ -n "$MAX_SAMPLES" ]]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
  fi

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    cmd+=(--skip-existing)
  fi

  "${cmd[@]}"
done
