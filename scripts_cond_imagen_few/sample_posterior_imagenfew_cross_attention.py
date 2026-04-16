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
    parser = argparse.ArgumentParser(
        description="Sample posterior generations from ImagenFewCrossAttention given input time series."
    )
    parser.add_argument("--config", type=str, required=True, help="Model config yaml.")
    parser.add_argument("--time-series-path", type=str, required=True, help="Path to input time series tensor (.pt or .npy).")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to finetuned ImagenFewCrossAttention checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Output path for posterior samples (.pt or .npy).")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name. If omitted, infer from config train_on_datasets when unique.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Real dataset split used for evaluation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of real samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size for sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--eval-metrics", nargs="+", type=str, default=None, help="Metrics to compute.")
    parser.add_argument("--ts2vec-dir", type=str, default=None, help="Directory for TS2Vec checkpoints/features.")
    parser.add_argument("--output-json", type=str, default=None, help="Where to save evaluation metrics json.")
    parser.add_argument("--save-generated-dir", type=str, default=None, help="Optional directory to save generated samples.")
    parser.add_argument("--no-ema-eval", action="store_true", help="Use raw model instead of EMA model.")
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
    args.eval_metrics = getattr(args, "eval_metrics", ["disc", "contextFID", "pred"])
    return args


def resolve_dataset_name(args, cli_dataset):
    if cli_dataset is not None:
        return cli_dataset
    if len(args.train_on_datasets) == 1:
        return args.train_on_datasets[0]
    raise ValueError("Please provide --dataset when config includes multiple train_on_datasets.")


def load_time_series(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Time series file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        array = np.load(path)
        tensor = torch.from_numpy(array)
    elif ext in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict):
            for key in ["samples", "data", "x", "time_series", "tensor"]:
                if key in loaded:
                    loaded = loaded[key]
                    break
        tensor = loaded if torch.is_tensor(loaded) else torch.as_tensor(loaded)
    else:
        raise ValueError("Unsupported time series file type. Please provide .pt/.pth or .npy")

    tensor = tensor.to(torch.float32)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"Expected time series tensor with shape (N, L, C) or (L, C), got {tuple(tensor.shape)}")
    return tensor


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


def save_output(path, tensor):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, tensor.cpu().numpy())
    else:
        torch.save(tensor.cpu(), path)


def make_default_output_json(dataset, split, model_ckpt, cond_path):
    results_dir = os.path.join("./logs", "conditional_imagen_few", "eval", dataset)
    os.makedirs(results_dir, exist_ok=True)
    model_tag = os.path.splitext(os.path.basename(model_ckpt))[0]
    cond_tag = os.path.splitext(os.path.basename(cond_path))[0]
    filename = f"{split}_posterior_{model_tag}__cond_{cond_tag}.json"
    return os.path.join(results_dir, filename)


def main():
    cli_args = parse_args()
    config = OmegaConf.to_object(OmegaConf.load(cli_args.config))
    args = to_args_namespace(config)

    args.model_ckpt = cli_args.model_ckpt
    if cli_args.batch_size is not None:
        args.batch_size = int(cli_args.batch_size)
    if cli_args.seed is not None:
        args.seed = int(cli_args.seed)
    if cli_args.eval_metrics is not None:
        args.eval_metrics = cli_args.eval_metrics

    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    class_label = dataset_list.index(dataset_name)

    torch.manual_seed(args.seed)
    np.random.default_rng(args.seed)

    condition_ts = load_time_series(cli_args.time_series_path)
    eval_tensor = build_eval_tensor(args, dataset_name, cli_args.split)
    eval_tensor = maybe_slice_tensor(eval_tensor, cli_args.max_samples)
    class_metadata = {
        "name": dataset_name,
        "channels": int(eval_tensor.shape[-1]),
    }

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()
    if cli_args.no_ema_eval and hasattr(handler, "_model") and hasattr(handler._model, "use_ema"):
        handler._model.use_ema = False

    with torch.no_grad():
        generated = handler._sample_with_posterior(
            n_samples=len(eval_tensor),
            class_metadata=class_metadata,
            test_data=condition_ts,
        )

    save_output(cli_args.output, generated)
    generated_np = generated.cpu().detach().numpy()
    real_np = eval_tensor.cpu().detach().numpy()

    ts2vec_dir = cli_args.ts2vec_dir or os.path.join("./logs", "TS2VEC")
    os.makedirs(ts2vec_dir, exist_ok=True)
    scores = evaluate_model_uncond(
        real_np,
        generated_np,
        dataset_name,
        args.device,
        args.eval_metrics,
        base_path=ts2vec_dir,
    )

    if cli_args.save_generated_dir:
        os.makedirs(cli_args.save_generated_dir, exist_ok=True)
        ext = os.path.splitext(cli_args.output)[1].lower() or ".pt"
        save_output(os.path.join(cli_args.save_generated_dir, f"{dataset_name}_{cli_args.split}_posterior{ext}"), generated)

    output_json = cli_args.output_json or make_default_output_json(
        dataset_name,
        cli_args.split,
        cli_args.model_ckpt,
        cli_args.time_series_path,
    )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    result = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "num_real_samples": int(len(eval_tensor)),
        "num_condition_samples": int(len(condition_ts)),
        "model_ckpt": cli_args.model_ckpt,
        "condition_time_series_path": cli_args.time_series_path,
        "generated_output_path": cli_args.output,
        "use_ema_for_eval": not cli_args.no_ema_eval,
        "metrics": {"posterior": scores},
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved posterior samples to {cli_args.output} with shape {tuple(generated.shape)}")
    print(json.dumps(result, indent=2))
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    main()
