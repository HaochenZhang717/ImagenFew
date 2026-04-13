import argparse
import json
import os
import sys
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.combined_datasets import dataset_list
from data_provider.data_provider import dataset_to_tensor, get_test, get_train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def to_args_namespace(config_dict):
    args = SimpleNamespace(**config_dict)
    args.ddp = False
    args.finetune = not getattr(args, "pretrain", False)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.n_classes = len(dataset_list)
    args.input_channels = getattr(args, "input_channels", None)
    args.subset_p = getattr(args, "subset_p", None)
    args.subset_n = getattr(args, "subset_n", None)
    args.find_unused_parameters = getattr(args, "find_unused_parameters", False)
    args.train_on_datasets = [dataset for dataset in dataset_list if dataset in args.train_on_datasets]
    args.learning_rate = getattr(args, "learning_rate", 1e-4)
    args.weight_decay = getattr(args, "weight_decay", 1e-5)
    args.epochs = getattr(args, "epochs", 1)
    args.warmup_steps = getattr(args, "warmup_steps", 0)
    args.min_lr = getattr(args, "min_lr", 0.0)
    args.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    args.seed = getattr(args, "seed", 42)
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


def summarize_latents(latents):
    flat = latents.reshape(latents.shape[0], -1).to(torch.float32)
    sample_norms = flat.norm(dim=1)
    dim_mean = flat.mean(dim=0)
    dim_std = flat.std(dim=0, unbiased=False)
    summary = {
        "shape": list(latents.shape),
        "global_mean": float(flat.mean().item()),
        "global_std": float(flat.std(unbiased=False).item()),
        "global_mean_abs": float(flat.abs().mean().item()),
        "global_rms": float(torch.sqrt((flat ** 2).mean()).item()),
        "sample_l2_mean": float(sample_norms.mean().item()),
        "sample_l2_std": float(sample_norms.std(unbiased=False).item()),
        "dim_mean_abs_avg": float(dim_mean.abs().mean().item()),
        "dim_std_avg": float(dim_std.mean().item()),
    }
    return summary


def compare_latents(posterior, generated):
    posterior_flat = posterior.reshape(posterior.shape[0], -1).to(torch.float32)
    generated_flat = generated.reshape(generated.shape[0], -1).to(torch.float32)

    posterior_mean = posterior_flat.mean(dim=0)
    generated_mean = generated_flat.mean(dim=0)
    posterior_std = posterior_flat.std(dim=0, unbiased=False)
    generated_std = generated_flat.std(dim=0, unbiased=False)

    mean_cosine = torch.nn.functional.cosine_similarity(
        posterior_mean.unsqueeze(0),
        generated_mean.unsqueeze(0),
        dim=1,
    ).item()

    return {
        "mean_vector_l2_diff": float((posterior_mean - generated_mean).norm().item()),
        "mean_vector_cosine": float(mean_cosine),
        "avg_abs_mean_diff_per_dim": float((posterior_mean - generated_mean).abs().mean().item()),
        "avg_abs_std_diff_per_dim": float((posterior_std - generated_std).abs().mean().item()),
        "global_mean_diff": float((posterior_flat.mean() - generated_flat.mean()).item()),
        "global_std_diff": float(
            (posterior_flat.std(unbiased=False) - generated_flat.std(unbiased=False)).item()
        ),
    }


def make_default_output_json(dataset, split):
    results_dir = os.path.join(
        "./logs",
        "self_conditional_generator",
        "latent_stats",
        dataset,
    )
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, f"{split}_prior_vs_posterior.json")


def main():
    cli_args = parse_args()
    config = OmegaConf.to_object(OmegaConf.load(cli_args.config))
    args = to_args_namespace(config)

    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    if cli_args.seed is not None:
        args.seed = cli_args.seed

    if not getattr(args, "prior_ckpt", None):
        raise ValueError("prior_ckpt must be set in the config for latent comparison.")

    torch.manual_seed(args.seed)
    np.random.default_rng(args.seed)

    eval_tensor = build_eval_tensor(args, dataset_name, cli_args.split)
    eval_tensor = maybe_slice_tensor(eval_tensor, cli_args.max_samples)

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()

    posterior_chunks = []
    with torch.no_grad():
        for start in range(0, len(eval_tensor), cli_args.batch_size):
            batch = eval_tensor[start:start + cli_args.batch_size].to(device=handler.device, dtype=torch.float32)
            posterior_chunks.append(handler._model.encode_posterior_context(batch).cpu())
    posterior_latents = torch.cat(posterior_chunks, dim=0)

    generated_latents = handler._draw_prior_contexts(len(eval_tensor)).cpu()

    result = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "num_samples": int(len(eval_tensor)),
        "prior_ckpt": args.prior_ckpt,
        "posterior": summarize_latents(posterior_latents),
        "generated": summarize_latents(generated_latents),
        "comparison": compare_latents(posterior_latents, generated_latents),
    }

    output_json = cli_args.output_json or make_default_output_json(dataset_name, cli_args.split)
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved latent statistics to {output_json}")


if __name__ == "__main__":
    main()
