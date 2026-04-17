# Trend-Then-Refine Pipeline

This note summarizes the current pipeline in this repo for:

1. generating a coarse `trend-only` time series, and then
2. refining that coarse series into a full-resolution time series.

The implementation is currently centered around:

- [ImagenFew trend-only training](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFew/handler.py)
- [ImagenFewRefine training and sampling](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFewRefine/handler.py)
- [run_refine.py](/Users/zhc/Documents/PhD/projects/ImagenFew/run_refine.py)

## 1. Stage A: Generate Trend-Only Time Series

The first stage trains a standard `ImagenFew` model, but with `trend_only=true`.

In this mode, the raw time series is preprocessed before being converted into an image:

1. Downsample with `AvgPool1d(kernel_size=2, stride=2)`
2. Upsample with `Upsample(scale_factor=2, mode='linear', align_corners=False)`

That logic lives in:

- [models/ImagenFew/handler.py](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFew/handler.py)

The same preprocessing is also applied during sampling and evaluation when `--trend_only` is passed:

- [scripts_imagen_few/sample_imagenfew.py](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_imagen_few/sample_imagenfew.py)

So this stage learns to generate a coarse, smoothed version of the original signal.

### Output of Stage A

The resulting generated trend-only samples are saved as `.npy` files, for example:

- `generated_ETTh2_train_trend_only.npy`
- `generated_AirQuality_train_trend_only.npy`
- `generated_mujoco_train_trend_only.npy`

These files are later used as one possible condition source for the refinement stage.

## 2. Stage B: Refine Coarse Trend into Full Time Series

The second stage uses `ImagenFewRefine`.

The intended role of `ImagenFewRefine` is:

- input: a coarse trend-like time series
- output: a refined full time series

This is implemented in:

- [models/ImagenFewRefine/handler.py](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFewRefine/handler.py)
- [models/ImagenFewRefine/networks.py](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFewRefine/networks.py)

## 3. Conditioning Interface in `ImagenFewRefine`

`ImagenFewRefine` conditions on a raw time-series tensor with shape:

- `(B, L, C)`

where:

- `B` = batch size
- `L` = sequence length
- `C` = number of channels

The conditioning signal is treated as a coarse trend signal.

### How the context is built

In [models/ImagenFewRefine/handler.py](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFewRefine/handler.py):

- `_build_trend_condition(x_ts)` applies the same `downsample + upsample` operation
- `_encode_context(trend_ts)` currently returns the trend tensor itself as `(B, L, C)`

Then, inside [models/ImagenFewRefine/networks.py](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFewRefine/networks.py), a small 1D CNN projector maps:

- `(B, L, C)` -> `(B, L_ctx, dim)`

The current `ContextProjector1D` uses three Conv1d layers and reduces sequence length while projecting the channel dimension into the cross-attention context space.

So the current refine model uses:

- raw coarse time series as input condition
- internal CNN projection to produce cross-attention tokens

## 4. Training Target Options

`run_refine.py` now supports two different training targets through:

- `--refine_target full`
- `--refine_target residual`

This argument is registered in:

- [utils/utils_args.py](/Users/zhc/Documents/PhD/projects/ImagenFew/utils/utils_args.py)

and consumed in:

- [run_refine.py](/Users/zhc/Documents/PhD/projects/ImagenFew/run_refine.py)
- [models/ImagenFewRefine/handler.py](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFewRefine/handler.py)

### `refine_target=full`

The model learns to directly generate the full time series.

Training target:

- `target_ts = x_ts`

### `refine_target=residual`

The model learns only the residual relative to the coarse trend.

Training target:

- `trend_ts = downsample_then_upsample(x_ts)`
- `target_ts = x_ts - trend_ts`

During sampling, the model first predicts a residual, and then the handler reconstructs the final output as:

- `full = predicted_residual + trend_condition`

This makes the residual mode a true coarse-to-detailed refinement formulation.

## 5. Sampling Modes in `ImagenFewRefine`

`ImagenFewRefine` currently supports three sampling modes via `sample_source`:

- `real_trend`
- `generated_trend`
- `both`

Implemented in:

- [models/ImagenFewRefine/handler.py](/Users/zhc/Documents/PhD/projects/ImagenFew/models/ImagenFewRefine/handler.py)

### `real_trend`

Uses real data from the evaluation dataset as the source of the trend condition:

1. Take the real full time series
2. Apply `downsample + upsample`
3. Use that coarse series as the conditioning signal
4. Generate a refined full series

### `generated_trend`

Uses externally generated trend-only samples as the source of the condition.

The path is passed through:

- `generated_trend_path`

The file can currently be:

- `.npy`
- `.pt`
- `.pth`

Expected shape:

- `(B, L, C)` or `(L, C)`

### `both`

Runs both of the above in the same evaluation:

- one refinement conditioned on real coarse trend
- one refinement conditioned on generated coarse trend

This is useful for directly measuring the gap between:

- ideal coarse trend conditioning
- actually generated trend conditioning

## 6. Current Finetune Configs

The current refine finetune configs are:

- [configs/refine/ETTh2.yaml](/Users/zhc/Documents/PhD/projects/ImagenFew/configs/refine/ETTh2.yaml)
- [configs/refine/AirQuality.yaml](/Users/zhc/Documents/PhD/projects/ImagenFew/configs/refine/AirQuality.yaml)
- [configs/refine/mujoco.yaml](/Users/zhc/Documents/PhD/projects/ImagenFew/configs/refine/mujoco.yaml)

These configs currently include:

- `sample_source: both`
- `generated_trend_path: ...`

so evaluation during refine finetuning compares:

- `real_trend`
- `generated_trend`

for each dataset.

## 7. Current Training Scripts

### Trend-only generation

The trend-only generation stage is launched from the `scripts_imagen_few` folder.

Examples:

- [scripts_imagen_few/imagen_few_trend_only_etth2_slurm.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_imagen_few/imagen_few_trend_only_etth2_slurm.sh)
- [scripts_imagen_few/submit_imagen_few_trend_only_finetune_slurm.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_imagen_few/submit_imagen_few_trend_only_finetune_slurm.sh)

### Refine finetuning

Refine finetuning is launched from the `scripts_refine` folder.

Examples:

- [scripts_refine/train_refine_etth2.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/train_refine_etth2.sh)
- [scripts_refine/train_refine_airquality.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/train_refine_airquality.sh)
- [scripts_refine/train_refine_mujoco.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/train_refine_mujoco.sh)
- [scripts_refine/submit_refine_finetune.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/submit_refine_finetune.sh)

Residual-target variants:

- [scripts_refine/train_refine_etth2_residual.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/train_refine_etth2_residual.sh)
- [scripts_refine/train_refine_airquality_residual.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/train_refine_airquality_residual.sh)
- [scripts_refine/train_refine_mujoco_residual.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/train_refine_mujoco_residual.sh)
- [scripts_refine/submit_refine_residual.sh](/Users/zhc/Documents/PhD/projects/ImagenFew/scripts_refine/submit_refine_residual.sh)

## 8. End-to-End Summary

The current pipeline is:

1. Train `ImagenFew` with `trend_only=true`
2. Generate and save trend-only time series samples
3. Train `ImagenFewRefine` using coarse trend as the condition
4. Choose whether the refine model predicts:
   - the full time series, or
   - only the residual over the coarse trend
5. During evaluation, compare:
   - refinement conditioned on real coarse trend
   - refinement conditioned on generated coarse trend

Conceptually:

```text
real full series
  -> downsample + upsample
  -> trend-only model training
  -> generated trend-only samples

generated trend-only samples OR real coarse trend
  -> ImagenFewRefine
  -> refined full time series
```

## 9. Practical Notes

- `ImagenFewRefine` still retains the dataset-token branch from the original `ImagenFew` backbone.
- The conditioning signal is no longer derived from VAE latents; it now comes from coarse time-series input directly.
- The context is projected internally by a small 1D CNN before cross-attention.
- If `refine_target=residual`, the final sampled output is automatically converted back to full time series by adding the coarse condition back in.

## 10. Suggested Future Cleanup

Some possible cleanup items if we want to make this pipeline even cleaner:

- expose `refine_target` explicitly in all refine config files
- add a dedicated evaluation script for refine-only experiments
- decide whether dataset-token conditioning should remain in `ImagenFewRefine`
- standardize naming between `real_trend / generated_trend / both` across scripts and config files
