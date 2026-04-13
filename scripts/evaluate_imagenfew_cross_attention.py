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
    parser.add_argument("--model-ckpt", type=str, default=None)
    parser.add_argument("--prior-ckpt", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--sample-source", type=str, default=None, choices=["prior", "posterior", "both"])
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
    args.beta1 = getattr(args, "beta1", 1e-5)
    args.betaT = getattr(args, "betaT", 1e-2)
    args.deterministic = getattr(args, "deterministic", False)
    args.input_channels = getattr(args, "input_channels", 1 if not getattr(args, "use_stft", False) else 2)
    args.lora_dim = getattr(args, "lora_dim", 4)
    args.dynamic_size = getattr(args, "dynamic_size", [128, 128])
    args.subset_p = getattr(args, "subset_p", None)
    args.subset_n = getattr(args, "subset_n", None)
    args.find_unused_parameters = getattr(args, "find_unused_parameters", False)
    args.train_on_datasets = [dataset for dataset in dataset_list if dataset in args.train_on_datasets]
    args.seed = getattr(args, "seed", 42)
    args.eval_metrics = getattr(args, "eval_metrics", ["disc_mean", "disc_std", "pred_mean", "pred_std", "context_fid"])
    args.context_dim = getattr(args, "context_dim", getattr(args, "multi_scale_vae", {}).get("z_channels"))
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


def make_default_output_json(dataset, split, sample_source, model_ckpt, prior_ckpt):
    results_dir = os.path.join("./logs", "conditional_imagen_few", "eval", dataset)
    os.makedirs(results_dir, exist_ok=True)
    model_tag = os.path.splitext(os.path.basename(model_ckpt))[0]
    prior_tag = os.path.splitext(os.path.basename(prior_ckpt))[0] if prior_ckpt else "no_prior"
    filename = f"{split}_{sample_source}_{model_tag}__{prior_tag}.json"
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
    if cli_args.model_ckpt is not None:
        args.model_ckpt = cli_args.model_ckpt
    if cli_args.prior_ckpt is not None:
        args.prior_ckpt = cli_args.prior_ckpt
    if cli_args.sample_source is not None:
        args.sample_source = cli_args.sample_source

    if not getattr(args, "model_ckpt", None):
        raise ValueError("A finetuned ImagenFewCrossAttention checkpoint must be provided via config or --model-ckpt.")
    if args.sample_source in {"prior", "both"} and not getattr(args, "prior_ckpt", None):
        raise ValueError("A diffusion prior checkpoint must be provided for prior sampling.")

    torch.manual_seed(args.seed)
    np.random.default_rng(args.seed)

    eval_tensor = build_eval_tensor(args, dataset_name, cli_args.split)
    eval_tensor = maybe_slice_tensor(eval_tensor, cli_args.max_samples)
    class_metadata = {"name": dataset_name, "channels": int(eval_tensor.shape[-1])}
    class_label = dataset_list.index(dataset_name)

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()
    if cli_args.no_ema_eval and hasattr(handler, "_model") and hasattr(handler._model, "use_ema"):
        handler._model.use_ema = False

    with torch.no_grad():
        generated_sets = handler.sample_variants(
            len(eval_tensor),
            class_label,
            class_metadata,
            eval_tensor,
        )

    real_set = eval_tensor.cpu().detach().numpy()
    ts2vec_dir = cli_args.ts2vec_dir or os.path.join("./logs", "TS2VEC")
    os.makedirs(ts2vec_dir, exist_ok=True)

    all_scores = {}
    saved_paths = {}
    for variant_name, generated_set in generated_sets.items():
        generated_np = generated_set.cpu().detach().numpy()
        scores = evaluate_model_uncond(
            real_set,
            generated_np,
            dataset_name,
            args.device,
            args.eval_metrics,
            base_path=ts2vec_dir,
        )
        all_scores[variant_name] = scores
        if cli_args.save_generated_dir:
            os.makedirs(cli_args.save_generated_dir, exist_ok=True)
            path = os.path.join(cli_args.save_generated_dir, f"{variant_name}.npy")
            np.save(path, generated_np)
            saved_paths[variant_name] = path

    output_json = cli_args.output_json or make_default_output_json(
        dataset_name,
        cli_args.split,
        args.sample_source,
        args.model_ckpt,
        getattr(args, "prior_ckpt", None),
    )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    result = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "num_samples": int(len(eval_tensor)),
        "sample_source": args.sample_source,
        "use_ema_for_eval": not cli_args.no_ema_eval,
        "model_ckpt": args.model_ckpt,
        "prior_ckpt": getattr(args, "prior_ckpt", None),
        "prior_sampler": getattr(args, "sampler", None),
        "metrics": all_scores,
        "generated_paths": saved_paths,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
