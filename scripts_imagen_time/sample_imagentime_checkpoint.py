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
from utils.utils_args import parse_args_uncond


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from an ImagenTime checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Path to ImagenTime config.")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to ImagenTime checkpoint (.pt).")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Which split decides the sample count.")
    parser.add_argument("--num-samples", type=int, default=None, help="If set, sample this many instead of matching the split size.")
    parser.add_argument("--batch-size", type=int, default=None, help="Sampling batch size override.")
    parser.add_argument(
        "--num-variants",
        type=int,
        default=10,
        help="How many independently sampled sets to generate and stack under sampled_ts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output .pt path.")
    parser.add_argument("--save-metadata", action="store_true", help="Save a small JSON next to the output.")
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
    args.train_on_datasets = [dataset for dataset in dataset_list if dataset in args.train_on_datasets]
    return args


def resolve_dataset_name(args, cli_dataset):
    if cli_dataset is not None:
        return cli_dataset
    if len(args.train_on_datasets) == 1:
        return args.train_on_datasets[0]
    raise ValueError("Please provide --dataset when the config includes multiple train_on_datasets.")


def build_split_tensor(args, dataset_name, split):
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


def default_output_path(model_ckpt, dataset_name, split, num_samples):
    ckpt_dir = os.path.dirname(os.path.abspath(model_ckpt))
    suffix = f"{split}_{num_samples}"
    return os.path.join(ckpt_dir, f"sampled_{dataset_name}_{suffix}.pt")


def main():
    cli_args = parse_args()
    args = load_training_args(cli_args.config)

    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    args.model_ckpt = cli_args.model_ckpt
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size

    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    ref_tensor = build_split_tensor(args, dataset_name, cli_args.split)
    ref_tensor = ref_tensor.detach().cpu()
    sample_count = cli_args.num_samples if cli_args.num_samples is not None else len(ref_tensor)
    class_label = dataset_list.index(dataset_name)
    class_metadata = {"name": dataset_name, "channels": int(ref_tensor.shape[-1])}

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()

    generated_variants = []
    with torch.no_grad():
        for variant_idx in range(cli_args.num_variants):
            current_seed = cli_args.seed + variant_idx
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            generated = handler.sample(sample_count, class_label, class_metadata).detach().cpu()
            if generated.shape[1] > generated.shape[2]:
                generated = generated.permute(0, 2, 1)
            generated_variants.append(generated)

    sampled_ts = torch.stack(generated_variants, dim=0)
    real_ts = ref_tensor[:sample_count]
    if real_ts.shape[1] > real_ts.shape[2]:
        real_ts = real_ts.permute(0, 2, 1)

    output_path = cli_args.output or default_output_path(
        cli_args.model_ckpt,
        dataset_name,
        cli_args.split,
        sample_count,
    )
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "real_ts": real_ts.float(),
        "sampled_ts": sampled_ts.float(),
        "dataset": dataset_name,
        "split": cli_args.split,
        "config": os.path.abspath(cli_args.config),
        "model_ckpt": os.path.abspath(cli_args.model_ckpt),
        "seed": int(cli_args.seed),
        "num_variants": int(cli_args.num_variants),
    }
    torch.save(payload, output_path)

    summary = {
        "config": os.path.abspath(cli_args.config),
        "model_ckpt": os.path.abspath(cli_args.model_ckpt),
        "dataset": dataset_name,
        "split": cli_args.split,
        "num_samples": int(sample_count),
        "num_variants": int(cli_args.num_variants),
        "batch_size": int(args.batch_size),
        "seed": int(cli_args.seed),
        "device": str(args.device),
        "output": output_path,
        "real_ts_shape": list(real_ts.shape),
        "sampled_ts_shape": list(sampled_ts.shape),
    }

    if cli_args.save_metadata:
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
