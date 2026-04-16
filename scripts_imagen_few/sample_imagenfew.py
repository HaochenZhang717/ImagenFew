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
from metrics import evaluate_model_uncond
from utils.utils_args import parse_args_uncond


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained ImagenFew checkpoint using an existing finetune config.")
    parser.add_argument("--config", type=str, required=True, help="Path to a config under configs/finetune.")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to the trained ImagenFew checkpoint (.pt).")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override. Defaults to the only train_on_datasets entry in the config.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to sample against for determining sample count and metadata.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of generated samples.")
    parser.add_argument("--output", type=str, default=None, help="Optional explicit .npy output path. Defaults to the checkpoint directory.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional explicit metrics JSON output path. Defaults to the checkpoint directory.")
    parser.add_argument("--eval-metrics", nargs="+", type=str, default=None, help="Metrics to compute. Defaults to the config's eval_metrics or disc/contextFID/pred.")
    parser.add_argument("--ts2vec-dir", type=str, default=None, help="Directory for TS2VEC checkpoints/cache used by contextFID.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--no-ema-eval", action="store_true", help="Disable EMA weights during sampling.")
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


def default_output_paths(model_ckpt, dataset_name, split):
    ckpt_dir = os.path.dirname(os.path.abspath(model_ckpt))
    npy_path = os.path.join(ckpt_dir, f"generated_{dataset_name}_{split}.npy")
    json_path = os.path.join(ckpt_dir, f"generated_{dataset_name}_{split}_eval.json")
    ts2vec_dir = os.path.join(ckpt_dir, "TS2VEC")
    return npy_path, json_path, ts2vec_dir


def main():
    cli_args = parse_args()
    args = load_training_args(cli_args.config)

    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    args.model_ckpt = cli_args.model_ckpt
    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.eval_metrics is not None:
        args.eval_metrics = cli_args.eval_metrics

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    eval_tensor = build_eval_tensor(args, dataset_name, cli_args.split)
    eval_tensor = maybe_slice_tensor(eval_tensor, cli_args.max_samples)
    class_metadata = {"name": dataset_name, "channels": int(eval_tensor.shape[-1])}
    class_label = dataset_list.index(dataset_name)

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()
    if cli_args.no_ema_eval and hasattr(handler, "_model") and hasattr(handler._model, "use_ema"):
        handler._model.use_ema = False

    eval_tensor = handler.preprocess_time_series(eval_tensor.to(args.device)).detach().cpu()

    with torch.no_grad():
        generated = handler.sample(
            len(eval_tensor),
            class_label,
            class_metadata,
            eval_tensor,
        )

    real_set = eval_tensor.cpu().detach().numpy()
    generated_np = generated.cpu().detach().numpy()

    output_path, summary_path, default_ts2vec_dir = default_output_paths(cli_args.model_ckpt, dataset_name, cli_args.split)
    if cli_args.output:
        output_path = os.path.abspath(cli_args.output)
    if cli_args.output_json:
        summary_path = os.path.abspath(cli_args.output_json)
    elif cli_args.output:
        summary_path = os.path.splitext(output_path)[0] + '_eval.json'

    ts2vec_dir = os.path.abspath(cli_args.ts2vec_dir) if cli_args.ts2vec_dir else default_ts2vec_dir

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    if 'contextFID' in args.eval_metrics:
        os.makedirs(ts2vec_dir, exist_ok=True)

    np.save(output_path, generated_np)
    scores = evaluate_model_uncond(
        real_set,
        generated_np,
        dataset_name,
        args.device,
        args.eval_metrics,
        base_path=ts2vec_dir if 'contextFID' in args.eval_metrics else None,
    )

    summary = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "num_samples": int(len(eval_tensor)),
        "config": os.path.abspath(cli_args.config),
        "model_ckpt": os.path.abspath(cli_args.model_ckpt),
        "generated_output": output_path,
        "evaluation_output": summary_path,
        "ts2vec_dir": ts2vec_dir if 'contextFID' in args.eval_metrics else None,
        "use_ema_for_eval": not cli_args.no_ema_eval,
        "seed": int(args.seed),
        "sample_shape": list(generated_np.shape),
        "eval_metrics": list(args.eval_metrics),
        "metrics": scores,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved generated samples to {output_path}")
    print(f"Saved evaluation results to {summary_path}")


if __name__ == "__main__":
    main()
