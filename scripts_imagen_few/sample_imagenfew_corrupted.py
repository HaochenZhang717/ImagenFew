import argparse
import json
import os
import sys
from importlib import import_module

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.combined_datasets import dataset_list
from data_provider.data_provider import dataset_to_tensor, get_test, get_train
from models.ImagenFew.sampler import DiffusionProcess
from utils.utils_args import parse_args_uncond


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate corrupted trend samples and reverse them with a trained ImagenFew generator."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to a finetune/pretrain config.")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to trained ImagenFew checkpoint (.pt).")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of real trends.")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output .npz path.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional explicit summary JSON path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--no-ema-eval", action="store_true", help="Disable EMA weights during reverse sampling.")
    parser.add_argument(
        "--trend_only",
        action="store_true",
        help="Apply downsample-then-upsample preprocessing before corruption, matching trend_only training.",
    )
    parser.add_argument(
        "--num-corruptions",
        type=int,
        default=5,
        help="Number of different corruption levels to sample for each real trend.",
    )
    parser.add_argument(
        "--min-step-frac",
        type=float,
        default=0.05,
        help="Minimum reverse-step fraction for sampling start times.",
    )
    parser.add_argument(
        "--max-step-frac",
        type=float,
        default=0.95,
        help="Maximum reverse-step fraction for sampling start times.",
    )
    return parser.parse_args()


def load_training_args(config_path):
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--config", config_path]
        args = parse_args_uncond()
    finally:
        sys.argv = old_argv

    args.ddp = False
    args.finetune = not getattr(args, "pretrain", False)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_classes = len(dataset_list)
    args.all_dataset_tokens = list(dataset_list)
    args.train_on_datasets = [dataset for dataset in dataset_list if dataset in args.train_on_datasets]
    return args


def resolve_dataset_name(args, cli_dataset):
    if cli_dataset is not None:
        return cli_dataset
    if len(args.train_on_datasets) == 1:
        return args.train_on_datasets[0]
    raise ValueError("Please provide --dataset when the config includes multiple train_on_datasets.")


def build_eval_tensor(args, dataset_name, split):
    dataset_config = None
    for config in args.datasets:
        if config["name"] == dataset_name:
            dataset_config = dict(config)
            break
    if dataset_config is None:
        raise ValueError(f"Dataset {dataset_name} not found in config.")

    dataset_config["seq_len"] = args.seq_len
    dataset_config["datasets_dir"] = args.datasets_dir
    dataset = get_train(dataset_config) if split == "train" else get_test(dataset_config)
    return dataset_to_tensor(dataset, args)


def maybe_slice_tensor(tensor, max_samples):
    if max_samples is None:
        return tensor
    return tensor[:max_samples]


def default_output_paths(model_ckpt, dataset_name, split, trend_only=False):
    ckpt_dir = os.path.dirname(os.path.abspath(model_ckpt))
    suffix = "_trend_only" if trend_only else ""
    npz_path = os.path.join(ckpt_dir, f"corrupted_{dataset_name}_{split}{suffix}.npz")
    json_path = os.path.join(ckpt_dir, f"corrupted_{dataset_name}_{split}{suffix}_summary.json")
    return npz_path, json_path


def get_t_steps(process, device):
    sigma_min = max(process.sigma_min, process.net.sigma_min)
    sigma_max = min(process.sigma_max, process.net.sigma_max)
    step_indices = torch.arange(process.num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / process.rho)
        + step_indices / (process.num_steps - 1) * (sigma_min ** (1 / process.rho) - sigma_max ** (1 / process.rho))
    ) ** process.rho
    return torch.cat([process.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])


def reverse_from_corrupted(process, x_t, mask, class_labels, start_index, t_steps):
    x_image_clear = torch.zeros_like(x_t, dtype=torch.float64)
    mask = mask.to(torch.float64)
    x_next = x_t.to(torch.float64) * (1 - mask) + x_image_clear

    for i in range(start_index, len(t_steps) - 1):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x_cur = x_next

        gamma = (
            min(process.S_churn / process.num_steps, np.sqrt(2) - 1)
            if process.S_min <= t_cur <= process.S_max
            else 0
        )
        t_hat = process.net.round_sigma(t_cur + gamma * t_cur)

        noise_step = (t_hat ** 2 - t_cur ** 2).sqrt() * process.S_noise * torch.randn_like(x_cur)
        noise_impute = noise_step * (1 - mask)
        x_to_impute = x_cur * (1 - mask) + noise_impute
        x_hat = x_image_clear + x_to_impute

        denoised = process.net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_image_clear + (x_hat + (t_next - t_hat) * d_cur) * (1 - mask)

        if i < len(t_steps) - 2:
            denoised = process.net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_image_clear + (x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)) * (1 - mask)

    return x_next.to(torch.float32)


def draw_step_indices(num_steps, num_corruptions, min_frac, max_frac, rng):
    if not (0.0 <= min_frac < max_frac <= 1.0):
        raise ValueError("--min-step-frac and --max-step-frac must satisfy 0 <= min < max <= 1.")

    max_positive_index = num_steps - 1
    min_index = max(0, int(np.floor(min_frac * max_positive_index)))
    max_index = min(max_positive_index - 1, int(np.ceil(max_frac * max_positive_index)))
    valid_indices = np.arange(min_index, max_index + 1)
    if len(valid_indices) == 0:
        raise ValueError("No valid corruption steps available. Please widen the step fraction range.")

    replace = len(valid_indices) < num_corruptions
    chosen = rng.choice(valid_indices, size=num_corruptions, replace=replace)
    chosen.sort()
    return chosen.astype(np.int64)


def main():
    cli_args = parse_args()
    args = load_training_args(cli_args.config)

    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    args.model_ckpt = cli_args.model_ckpt
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    args.trend_only = bool(cli_args.trend_only or getattr(args, "trend_only", False))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    eval_tensor = build_eval_tensor(args, dataset_name, cli_args.split)
    eval_tensor = maybe_slice_tensor(eval_tensor, cli_args.max_samples)
    class_metadata = {"name": dataset_name, "channels": int(eval_tensor.shape[-1])}
    class_label = dataset_list.index(dataset_name)

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()
    if cli_args.no_ema_eval and hasattr(handler, "_model") and hasattr(handler._model, "use_ema"):
        handler._model.use_ema = False

    output_path, summary_path = default_output_paths(
        cli_args.model_ckpt,
        dataset_name,
        cli_args.split,
        trend_only=args.trend_only,
    )
    if cli_args.output:
        output_path = os.path.abspath(cli_args.output)
    if cli_args.output_json:
        summary_path = os.path.abspath(cli_args.output_json)
    elif cli_args.output:
        summary_path = os.path.splitext(output_path)[0] + "_summary.json"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with torch.no_grad():
        real_trend = handler.preprocess_time_series(eval_tensor.to(args.device)).detach()
        process = DiffusionProcess(
            args,
            handler._model.net,
            (class_metadata["channels"], args.img_resolution, args.img_resolution),
        )
        t_steps = get_t_steps(process, args.device)

        batch_size = args.batch_size
        num_samples = len(real_trend)
        num_corruptions = cli_args.num_corruptions

        corrupted_ts_all = []
        reconstructed_ts_all = []
        step_indices_all = []
        sigma_all = []

        scope = handler._model.ema_scope if hasattr(handler, "_model") else torch.no_grad
        with scope():
            for start in range(0, num_samples, batch_size):
                batch = real_trend[start:start + batch_size]
                batch_len = batch.shape[0]
                clean_img = handler._model.ts_to_img(batch)
                img_mask = handler._model.ts_to_img(torch.zeros_like(batch), pad_val=1).to(torch.float32)
                oh_class_labels = torch.nn.functional.one_hot(
                    torch.full((batch_len,), class_label, device=args.device),
                    num_classes=args.n_classes,
                ).to(torch.float32)

                batch_corrupted = []
                batch_reconstructed = []
                batch_step_indices = []
                batch_sigmas = []

                for sample_idx in range(batch_len):
                    chosen_step_indices = draw_step_indices(
                        args.diffusion_steps,
                        num_corruptions,
                        cli_args.min_step_frac,
                        cli_args.max_step_frac,
                        rng,
                    )
                    sample_sigmas = t_steps[chosen_step_indices].detach().cpu().numpy().astype(np.float32)

                    sample_clean_img = clean_img[sample_idx:sample_idx + 1]
                    sample_mask = img_mask[sample_idx:sample_idx + 1]
                    sample_labels = oh_class_labels[sample_idx:sample_idx + 1]

                    corrupted_ts_variants = []
                    reconstructed_ts_variants = []

                    for step_index in chosen_step_indices:
                        sigma = t_steps[int(step_index)].to(torch.float32)
                        noise = torch.randn_like(sample_clean_img)
                        corrupted_img = (sample_clean_img + sigma * noise) * (1 - sample_mask)
                        reconstructed_img = reverse_from_corrupted(
                            process,
                            corrupted_img,
                            sample_mask,
                            sample_labels,
                            int(step_index),
                            t_steps,
                        )

                        corrupted_ts = handler._model.img_to_ts(corrupted_img)[:, :, :class_metadata["channels"]]
                        reconstructed_ts = handler._model.img_to_ts(reconstructed_img)[:, :, :class_metadata["channels"]]
                        corrupted_ts_variants.append(corrupted_ts.squeeze(0).detach().cpu())
                        reconstructed_ts_variants.append(reconstructed_ts.squeeze(0).detach().cpu())

                    batch_corrupted.append(torch.stack(corrupted_ts_variants, dim=0))
                    batch_reconstructed.append(torch.stack(reconstructed_ts_variants, dim=0))
                    batch_step_indices.append(torch.from_numpy(chosen_step_indices))
                    batch_sigmas.append(torch.from_numpy(sample_sigmas))

                corrupted_ts_all.append(torch.stack(batch_corrupted, dim=0))
                reconstructed_ts_all.append(torch.stack(batch_reconstructed, dim=0))
                step_indices_all.append(torch.stack(batch_step_indices, dim=0))
                sigma_all.append(torch.stack(batch_sigmas, dim=0))

    corrupted_ts = torch.cat(corrupted_ts_all, dim=0).numpy()
    reconstructed_ts = torch.cat(reconstructed_ts_all, dim=0).numpy()
    sampled_step_indices = torch.cat(step_indices_all, dim=0).numpy()
    sampled_sigmas = torch.cat(sigma_all, dim=0).numpy()
    real_trend_np = real_trend.detach().cpu().numpy()

    corruption_mse = ((corrupted_ts - real_trend_np[:, None, :, :]) ** 2).mean(axis=(2, 3))
    reconstruction_mse = ((reconstructed_ts - real_trend_np[:, None, :, :]) ** 2).mean(axis=(2, 3))
    reconstruction_mae = np.abs(reconstructed_ts - real_trend_np[:, None, :, :]).mean(axis=(2, 3))

    np.savez_compressed(
        output_path,
        real_trend=real_trend_np,
        corrupted_trend=corrupted_ts,
        reconstructed_trend=reconstructed_ts,
        step_indices=sampled_step_indices,
        sigmas=sampled_sigmas,
    )

    summary = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "num_samples": int(real_trend_np.shape[0]),
        "num_corruptions_per_sample": int(num_corruptions),
        "config": os.path.abspath(cli_args.config),
        "model_ckpt": os.path.abspath(cli_args.model_ckpt),
        "output": output_path,
        "summary_json": summary_path,
        "use_ema_for_eval": not cli_args.no_ema_eval,
        "trend_only": bool(args.trend_only),
        "seed": int(args.seed),
        "sample_shape": {
            "real_trend": list(real_trend_np.shape),
            "corrupted_trend": list(corrupted_ts.shape),
            "reconstructed_trend": list(reconstructed_ts.shape),
        },
        "step_sampling": {
            "min_step_frac": float(cli_args.min_step_frac),
            "max_step_frac": float(cli_args.max_step_frac),
            "step_index_min": int(sampled_step_indices.min()),
            "step_index_max": int(sampled_step_indices.max()),
            "sigma_min": float(sampled_sigmas.min()),
            "sigma_max": float(sampled_sigmas.max()),
        },
        "metrics": {
            "corruption_mse_mean": float(corruption_mse.mean()),
            "corruption_mse_std": float(corruption_mse.std()),
            "reconstruction_mse_mean": float(reconstruction_mse.mean()),
            "reconstruction_mse_std": float(reconstruction_mse.std()),
            "reconstruction_mae_mean": float(reconstruction_mae.mean()),
            "reconstruction_mae_std": float(reconstruction_mae.std()),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved corrupted/reconstructed trends to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
