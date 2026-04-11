"""
Export finetune datasets to separate .npy files with shape (N, T, C).

By default this script reads all yaml files from configs/self_cond_finetune and
saves one .npy per dataset.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from data_provider.combined_datasets import dataset_list
from data_provider.data_provider import data_provider, dataset_to_tensor, get_test, get_train


def load_config(config_path):
    return OmegaConf.to_object(OmegaConf.load(config_path))


def build_tensor_args(cfg, batch_size):
    return SimpleNamespace(
        seq_len=cfg["seq_len"],
        batch_size=batch_size,
    )


def export_single_dataset(cfg, split, batch_size):
    dataset_cfg = None
    for candidate in cfg["datasets"]:
        if candidate["name"] in cfg["train_on_datasets"]:
            dataset_cfg = dict(candidate)
            break

    if dataset_cfg is None:
        raise ValueError(f"Could not find a dataset matching train_on_datasets={cfg['train_on_datasets']}")

    dataset_cfg["seq_len"] = cfg["seq_len"]
    dataset_cfg["datasets_dir"] = cfg["datasets_dir"]

    if split == "train":
        dataset = get_train(dataset_cfg)
    elif split == "test":
        dataset = get_test(dataset_cfg)
    else:
        raise ValueError(f"Unsupported split: {split}")

    tensor = dataset_to_tensor(dataset, build_tensor_args(cfg, batch_size))
    return dataset_cfg["name"], tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-config-dir",
        type=str,
        default="./configs/self_cond_finetune",
        help="Directory containing finetune yaml configs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./logs/finetune_datasets_npy",
        help="Directory to save one .npy file per dataset",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_paths = sorted(
        os.path.join(args.dataset_config_dir, name)
        for name in os.listdir(args.dataset_config_dir)
        if name.endswith(".yaml")
    )

    saved = {}
    for config_path in config_paths:
        cfg = load_config(config_path)
        cfg["train_on_datasets"] = [
            dataset for dataset in dataset_list if dataset in cfg["train_on_datasets"]
        ]
        dataset_name, tensor = export_single_dataset(
            cfg=cfg,
            split=args.split,
            batch_size=args.batch_size,
        )
        output_path = os.path.join(args.output_dir, f"{dataset_name}_{args.split}.npy")
        np.save(output_path, tensor.cpu().numpy())
        saved[dataset_name] = {
            "path": output_path,
            "shape": tuple(tensor.shape),
            "split": args.split,
        }
        print(f"Saved {dataset_name} ({args.split}) to {output_path} with shape {tuple(tensor.shape)}")

    index_path = os.path.join(args.output_dir, f"index_{args.split}.pt")
    torch.save(saved, index_path)
    print(f"Saved export index to {index_path}")


if __name__ == "__main__":
    main()
