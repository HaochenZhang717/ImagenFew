"""
Plot every sample/channel pair from datasets defined in a config file.

This version is streaming-friendly: it does not first materialize the whole
dataset as a giant tensor in memory.

Examples:
    python scripts/plot_dataset_channels.py \
        --config ./configs/self_cond_finetune/ETTh2.yaml \
        --split train \
        --max-samples 100

    python scripts/plot_dataset_channels.py \
        --config ./configs/pretrain/self_conditional.yaml \
        --dataset Weather \
        --split test \
        --output-dir ./plots/weather_test \
        --skip-existing
"""

import argparse
import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import Subset
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from data_provider.data_provider import get_test, get_train
from utils.utils_vis import sample_to_numpy_2d, save_one_sample_channel_plots


def load_config(config_path, overrides):
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return OmegaConf.to_object(cfg)


def build_dataset(dataset_cfg, global_cfg, split, max_samples=None):
    config = copy.deepcopy(dataset_cfg)
    config["seq_len"] = global_cfg["seq_len"]
    config["datasets_dir"] = global_cfg["datasets_dir"]

    if split == "train":
        dataset = get_train(config)
    elif split == "test":
        dataset = get_test(config)
    else:
        raise ValueError(f"Unsupported split: {split}")

    if max_samples is not None:
        max_samples = min(int(max_samples), len(dataset))
        dataset = Subset(dataset, torch.arange(max_samples))

    return dataset


def select_datasets(cfg, dataset_name=None):
    datasets = cfg["datasets"]
    if dataset_name is None:
        train_on_datasets = cfg.get("train_on_datasets")
        if train_on_datasets is None:
            return datasets
        selected = [dataset for dataset in datasets if dataset["name"] in train_on_datasets]
        if selected:
            return selected
        return datasets

    selected = [dataset for dataset in datasets if dataset["name"] == dataset_name]
    if not selected:
        available = ", ".join(dataset["name"] for dataset in datasets)
        raise ValueError(f"Dataset {dataset_name} not found in config. Available: {available}")
    return selected


def sample_dir_exists(save_dir, prefix, sample_idx):
    return (Path(save_dir) / f"{prefix}_{sample_idx:05d}").exists()


def count_pending_samples(dataset_length, save_dir, prefix, skip_existing):
    if not skip_existing:
        return dataset_length
    pending = 0
    for sample_idx in range(dataset_length):
        if not sample_dir_exists(save_dir, prefix, sample_idx):
            pending += 1
    return pending


def save_one_sample_channel_plots_worker(sample, sample_idx, save_dir, dpi, figsize, prefix):
    save_one_sample_channel_plots(
        sample=sample,
        sample_idx=sample_idx,
        save_dir=save_dir,
        dpi=dpi,
        figsize=figsize,
        prefix=prefix,
    )
    return sample_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset name inside config")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output-dir", type=str, default="./plots/dataset_channels")
    parser.add_argument("--max-samples", type=int, default=None, help="Only plot the first N samples")
    parser.add_argument("--prefix", type=str, default="sample", help="Filename prefix for saved samples")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fig-width", type=float, default=8.0)
    parser.add_argument("--fig-height", type=float, default=3.0)
    parser.add_argument("--num-workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples whose output directory already exists")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    selected_datasets = select_datasets(cfg, args.dataset)
    figsize = (args.fig_width, args.fig_height)

    for dataset_cfg in selected_datasets:
        dataset_name = dataset_cfg["name"]
        dataset = build_dataset(
            dataset_cfg=dataset_cfg,
            global_cfg=cfg,
            split=args.split,
            max_samples=args.max_samples,
        )
        save_dir = os.path.join(args.output_dir, dataset_name, args.split)
        os.makedirs(save_dir, exist_ok=True)
        pending_samples = count_pending_samples(len(dataset), save_dir, args.prefix, args.skip_existing)
        print(
            f"Saving plots for {dataset_name} ({args.split}) to {save_dir} from {len(dataset)} samples "
            f"(pending={pending_samples}, workers={args.num_workers})"
        )

        if pending_samples == 0:
            continue

        if args.num_workers <= 1:
            for sample_idx in tqdm(range(len(dataset)), total=pending_samples, desc=f"{dataset_name}-{args.split}", leave=False):
                if args.skip_existing and sample_dir_exists(save_dir, args.prefix, sample_idx):
                    continue
                save_one_sample_channel_plots(
                    sample=dataset[sample_idx],
                    sample_idx=sample_idx,
                    save_dir=save_dir,
                    dpi=args.dpi,
                    figsize=figsize,
                    prefix=args.prefix,
                )
            continue

        mp_context = get_context("spawn")
        max_pending_futures = max(1, args.num_workers * 2)
        pending_futures = []

        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context) as executor:
            with tqdm(total=pending_samples, desc=f"{dataset_name}-{args.split}", leave=False) as pbar:
                for sample_idx in range(len(dataset)):
                    if args.skip_existing and sample_dir_exists(save_dir, args.prefix, sample_idx):
                        continue

                    sample = sample_to_numpy_2d(dataset[sample_idx])
                    future = executor.submit(
                        save_one_sample_channel_plots_worker,
                        sample,
                        sample_idx,
                        save_dir,
                        args.dpi,
                        figsize,
                        args.prefix,
                    )
                    pending_futures.append(future)

                    if len(pending_futures) >= max_pending_futures:
                        pending_futures[0].result()
                        pending_futures.pop(0)
                        pbar.update(1)

                while pending_futures:
                    pending_futures[0].result()
                    pending_futures.pop(0)
                    pbar.update(1)


if __name__ == "__main__":
    main()
