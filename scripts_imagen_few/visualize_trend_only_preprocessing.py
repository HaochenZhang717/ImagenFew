import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.data_provider import dataset_to_tensor, get_test, get_train
from utils.utils_args import parse_args_uncond


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize downsample+upsample trend-only preprocessing on time series data.")
    parser.add_argument("--config", type=str, required=True, help="Path to an ImagenFew config.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override. Defaults to the only train_on_datasets entry in the config.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Which split to visualize.")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of the sample to visualize.")
    parser.add_argument("--channel-idx", type=int, default=0, help="Channel index to visualize.")
    parser.add_argument("--all-channels", action="store_true", help="Visualize all channels for the selected sample.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the figure.")
    parser.add_argument("--stats-json", type=str, default=None, help="Optional path to save summary statistics as JSON.")
    return parser.parse_args()


def load_training_args(config_path):
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--config", config_path]
        args = parse_args_uncond()
    finally:
        sys.argv = old_argv

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def resolve_dataset_name(args, cli_dataset):
    if cli_dataset is not None:
        return cli_dataset
    if len(args.train_on_datasets) == 1:
        return args.train_on_datasets[0]
    raise ValueError("Please provide --dataset when the config includes multiple train_on_datasets.")


def build_tensor(args, dataset_name, split):
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


def preprocess_trend_only(x_ts):
    original_len = x_ts.shape[1]
    x_ts = x_ts.permute(0, 2, 1)
    x_ts = F.avg_pool1d(x_ts, kernel_size=2, stride=2)
    x_ts = F.interpolate(x_ts, scale_factor=2, mode="linear", align_corners=False)
    if x_ts.shape[-1] != original_len:
        x_ts = F.interpolate(x_ts, size=original_len, mode="linear", align_corners=False)
    return x_ts.permute(0, 2, 1).contiguous()


def main():
    cli_args = parse_args()
    args = load_training_args(cli_args.config)
    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    tensor = build_tensor(args, dataset_name, cli_args.split)

    if cli_args.sample_idx < 0 or cli_args.sample_idx >= tensor.shape[0]:
        raise IndexError(f"sample_idx {cli_args.sample_idx} is out of range for dataset size {tensor.shape[0]}.")
    if not cli_args.all_channels and (cli_args.channel_idx < 0 or cli_args.channel_idx >= tensor.shape[-1]):
        raise IndexError(f"channel_idx {cli_args.channel_idx} is out of range for channel size {tensor.shape[-1]}.")

    trend_tensor = preprocess_trend_only(tensor)
    channel_indices = list(range(tensor.shape[-1])) if cli_args.all_channels else [cli_args.channel_idx]
    original_all = tensor[cli_args.sample_idx].cpu().numpy()
    trend_all = trend_tensor[cli_args.sample_idx].cpu().numpy()
    residual_all = original_all - trend_all

    per_channel_stats = []
    for channel_idx in channel_indices:
        residual = residual_all[:, channel_idx]
        per_channel_stats.append(
            {
                "channel_idx": int(channel_idx),
                "mse": float((residual ** 2).mean()),
                "mae": float(abs(residual).mean()),
                "max_abs_diff": float(abs(residual).max()),
            }
        )

    summary = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "sample_idx": int(cli_args.sample_idx),
        "all_channels": bool(cli_args.all_channels),
        "num_channels": int(tensor.shape[-1]),
        "seq_len": int(original_all.shape[0]),
        "global_mse": float((residual_all ** 2).mean()),
        "global_mae": float(abs(residual_all).mean()),
        "global_max_abs_diff": float(abs(residual_all).max()),
        "per_channel": per_channel_stats,
    }

    n_channels = len(channel_indices)
    fig_height = max(4.5, 2.8 * n_channels)
    fig, axes = plt.subplots(n_channels, 2, figsize=(14, fig_height), sharex="col", squeeze=False)
    fig.suptitle(
        f"{dataset_name} | {cli_args.split} | sample={cli_args.sample_idx} | "
        f"{'all channels' if cli_args.all_channels else f'channel={cli_args.channel_idx}'}",
        fontsize=14,
    )

    for row, channel_idx in enumerate(channel_indices):
        original = original_all[:, channel_idx]
        trend = trend_all[:, channel_idx]
        residual = residual_all[:, channel_idx]
        channel_stats = per_channel_stats[row]

        axes[row, 0].plot(original, label="original", linewidth=1.8)
        axes[row, 0].plot(trend, label="downsample+upsample", linewidth=1.8)
        axes[row, 0].set_title(f"Channel {channel_idx}")
        axes[row, 0].grid(alpha=0.3)
        if row == 0:
            axes[row, 0].legend()

        axes[row, 1].plot(residual, color="tab:red", label="residual", linewidth=1.8)
        axes[row, 1].set_title(
            f"Residual | MSE={channel_stats['mse']:.5f} "
            f"MAE={channel_stats['mae']:.5f} "
            f"MaxAbs={channel_stats['max_abs_diff']:.5f}"
        )
        axes[row, 1].grid(alpha=0.3)
        if row == 0:
            axes[row, 1].legend()

    axes[-1, 0].set_xlabel("time step")
    axes[-1, 1].set_xlabel("time step")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    print(json.dumps(summary, indent=2))

    if cli_args.save is not None:
        save_path = os.path.abspath(cli_args.save)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    if cli_args.stats_json is not None:
        stats_path = os.path.abspath(cli_args.stats_json)
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved statistics to {stats_path}")

    if cli_args.save is None:
        plt.show()


if __name__ == "__main__":
    main()
