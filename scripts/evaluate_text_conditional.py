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
from metrics import evaluate_model_uncond


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--conditional-ckpt", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--save-generated-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-metrics", nargs="+", type=str, default=None)
    parser.add_argument("--ts2vec-dir", type=str, default=None)
    parser.add_argument("--no-ema-eval", action="store_true")
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
    args.sample_source = "posterior"
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
    if split == "train":
        dataset = get_train(dataset_config)
    else:
        dataset = get_test(dataset_config)
    return dataset_to_tensor(dataset, args)


def maybe_slice_tensor(tensor, max_samples):
    if max_samples is None:
        return tensor
    return tensor[:max_samples]


def load_caption_embeddings(path):
    embeds = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(embeds):
        raise ValueError(f"Expected caption embeddings tensor at {path}, got {type(embeds)}")
    if embeds.ndim != 3:
        raise ValueError(f"Expected caption embeddings with shape (N, n_var, dim), got {tuple(embeds.shape)}")
    return embeds.to(torch.float32)


def make_default_output_json(dataset, split, conditional_ckpt):
    results_dir = os.path.join(
        "./logs",
        "text_conditional_generator",
        "eval",
        dataset,
    )
    os.makedirs(results_dir, exist_ok=True)
    cond_tag = os.path.splitext(os.path.basename(conditional_ckpt))[0]
    filename = f"{split}_posterior_{cond_tag}.json"
    return os.path.join(results_dir, filename)


def main():
    cli_args = parse_args()
    config = OmegaConf.to_object(OmegaConf.load(cli_args.config))
    args = to_args_namespace(config)

    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.eval_metrics is not None:
        args.eval_metrics = cli_args.eval_metrics
    if cli_args.conditional_ckpt is not None:
        args.resume_ckpt = cli_args.conditional_ckpt

    if not getattr(args, "resume_ckpt", None):
        raise ValueError("A text-conditional checkpoint must be provided via config or --conditional-ckpt.")
    if not getattr(args, "caption_embeddings_path", None):
        raise ValueError("caption_embeddings_path must be provided in the config.")

    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)

    eval_tensor = build_eval_tensor(args, dataset_name, cli_args.split)
    caption_embeddings = load_caption_embeddings(args.caption_embeddings_path)

    if len(eval_tensor) != len(caption_embeddings):
        raise ValueError(
            f"Time-series split length {len(eval_tensor)} does not match caption embeddings length {len(caption_embeddings)}. "
            f"Current split={cli_args.split}, embeddings={args.caption_embeddings_path}"
        )

    eval_tensor = maybe_slice_tensor(eval_tensor, cli_args.max_samples)
    caption_embeddings = maybe_slice_tensor(caption_embeddings, cli_args.max_samples)

    class_metadata = {"name": dataset_name, "channels": int(eval_tensor.shape[-1])}
    class_label = dataset_list.index(dataset_name)

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()
    if cli_args.no_ema_eval and hasattr(handler, "_model") and hasattr(handler._model, "use_ema"):
        handler._model.use_ema = False

    with torch.no_grad():
        generated = handler.sample(
            len(eval_tensor),
            class_label,
            class_metadata,
            (eval_tensor, caption_embeddings),
        )

    real_set = eval_tensor.cpu().detach().numpy()
    generated_np = generated.cpu().detach().numpy()
    ts2vec_dir = cli_args.ts2vec_dir or os.path.join("./logs", "TS2VEC")
    os.makedirs(ts2vec_dir, exist_ok=True)

    scores = evaluate_model_uncond(
        real_set,
        generated_np,
        dataset_name,
        args.device,
        args.eval_metrics,
        args.metric_iteration,
        ts2vec_dir,
    )

    if cli_args.save_generated_dir:
        os.makedirs(cli_args.save_generated_dir, exist_ok=True)
        generated_path = os.path.join(
            cli_args.save_generated_dir,
            f"{dataset_name}_{cli_args.split}_posterior_samples.npy",
        )
        np.save(generated_path, generated_np)
    else:
        generated_path = None

    output_json = cli_args.output_json or make_default_output_json(
        dataset_name,
        cli_args.split,
        args.resume_ckpt,
    )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    result = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "conditional_ckpt": args.resume_ckpt,
        "caption_embeddings_path": args.caption_embeddings_path,
        "use_ema_for_eval": not cli_args.no_ema_eval,
        "num_samples": int(len(eval_tensor)),
        "metrics": scores,
        "generated_path": generated_path,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
