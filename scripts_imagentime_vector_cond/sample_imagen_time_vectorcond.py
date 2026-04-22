import argparse
import json
import os
import re
import sys
from importlib import import_module

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.combined_datasets import dataset_list
from data_provider.data_provider import data_provider, dataset_to_tensor, get_test, get_train
from metrics import evaluate_model_uncond
from utils.utils import create_model_name_and_dir
from utils.utils_args import parse_args_uncond


EPOCH_PATTERN = re.compile(r"epoch_(\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load an ImagenTimeVectorCond checkpoint and sample from a specified embedding tensor."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to ImagenTimeVectorCond config.")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to checkpoint (.pt).")
    parser.add_argument("--embeds-path", type=str, required=True, help="Path to conditioning embeddings (.pt).")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument(
        "--reference-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split used for metadata, scaler state, and optional real_ts saving.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="If set, truncate the embedding bank.")
    parser.add_argument("--batch-size", type=int, default=None, help="Sampling batch size override.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Optional explicit output .pt path.")
    parser.add_argument("--save-metadata", action="store_true", help="Save a small JSON next to the output.")
    parser.add_argument("--eval-metrics", nargs="+", default=["disc", "contextFID", "pred", "vaeFID"])
    parser.add_argument("--metric-iteration", type=int, default=10)
    parser.add_argument("--ts2vec-dir", type=str, default=None)
    parser.add_argument("--fid-vae-ckpt-root", type=str, default=None)
    parser.add_argument("--output-jsonl", type=str, default=None, help="Append one metric summary record to this jsonl file.")
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


def get_dataset_config(args, dataset_name):
    for config in args.datasets:
        if config["name"] == dataset_name:
            dataset_config = dict(config)
            dataset_config["seq_len"] = args.seq_len
            dataset_config["datasets_dir"] = args.datasets_dir
            return dataset_config
    raise ValueError(f"Dataset {dataset_name} not found in config.")


def build_split_tensor(args, dataset_name, split):
    dataset_config = get_dataset_config(args, dataset_name)
    dataset = get_train(dataset_config) if split == "train" else get_test(dataset_config)
    return dataset_to_tensor(dataset, args)


def load_condition_embeddings(path):
    embeds = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(embeds, dict):
        if "embeddings" in embeds:
            embeds = embeds["embeddings"]
        else:
            raise ValueError(f"Expected embedding dict at {path} to contain 'embeddings'.")
    if not torch.is_tensor(embeds):
        raise TypeError(f"Expected torch.Tensor embeddings at {path}, got {type(embeds)}")
    if embeds.ndim == 3:
        if embeds.shape[1] != 1:
            raise ValueError(
                f"Expected 2D embeddings or singleton-token 3D embeddings, got shape {tuple(embeds.shape)}"
            )
        embeds = embeds[:, 0, :]
    if embeds.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (N, D), got {tuple(embeds.shape)}")
    return embeds.to(torch.float32)


def maybe_slice_tensor(tensor, max_samples):
    if max_samples is None:
        return tensor
    return tensor[:max_samples]


def _extract_real_tensor(dataset):
    if hasattr(dataset, "tensors"):
        return dataset.tensors[0]
    if isinstance(dataset, (tuple, list)):
        return dataset[0]
    return dataset


def _save_eval_samples(output_path, payload):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)


def _to_ntc(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        x = tensor_or_array.detach().cpu().float()
    else:
        x = torch.as_tensor(tensor_or_array).detach().cpu().float()

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor with shape (N, T, C) or (N, C, T), got {tuple(x.shape)}")
    if x.shape[1] < x.shape[2]:
        x = x.permute(0, 2, 1)
    return x.numpy()


def _extract_epoch_from_ckpt(model_ckpt):
    match = EPOCH_PATTERN.search(os.path.basename(model_ckpt))
    return int(match.group(1)) if match else -1


def default_output_path(args, dataset_name, embeds_path):
    samples_dir = os.path.join(os.path.dirname(args.log_dir), "eval_samples")
    os.makedirs(samples_dir, exist_ok=True)
    embed_tag = os.path.splitext(os.path.basename(embeds_path))[0]
    ckpt_tag = os.path.splitext(os.path.basename(args.model_ckpt))[0]
    return os.path.join(samples_dir, f"{dataset_name}_{embed_tag}_{ckpt_tag}.pt")


def main():
    cli_args = parse_args()
    args = load_training_args(cli_args.config)

    dataset_name = resolve_dataset_name(args, cli_args.dataset)
    args.model_ckpt = cli_args.model_ckpt
    args.dataset = dataset_name
    if cli_args.batch_size is not None:
        args.batch_size = cli_args.batch_size

    torch.manual_seed(cli_args.seed)
    np.random.seed(cli_args.seed)

    # Align argument initialization and metadata construction with training entrypoints.
    create_model_name_and_dir(args)
    _, _, trainsets, metadatas = data_provider(args)
    if dataset_name not in metadatas:
        raise ValueError(f"Dataset {dataset_name} was not prepared by data_provider.")

    reference_dataset = trainsets[dataset_name] if cli_args.reference_split == "train" else build_split_tensor(args, dataset_name, "test")
    reference_tensor = _extract_real_tensor(reference_dataset).detach().cpu()
    embeddings = maybe_slice_tensor(load_condition_embeddings(cli_args.embeds_path), cli_args.max_samples)
    if len(reference_tensor) != len(embeddings):
        raise ValueError(
            f"reference split size ({len(reference_tensor)}) does not match embedding count ({len(embeddings)})."
        )

    expected_dim = int(getattr(args, "condition_dim", getattr(args, "context_dim", embeddings.shape[-1])))
    if embeddings.shape[-1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: checkpoint/config expects {expected_dim}, got {embeddings.shape[-1]}"
        )

    class_label = dataset_list.index(dataset_name)
    class_metadata = dict(metadatas[dataset_name])
    sample_count = len(embeddings)

    handler = import_module(args.handler).Handler(args=args, rank=args.device)
    handler.model.eval()

    # Delay embedding caches the image geometry during ts_to_img().
    # Training initializes this naturally, but checkpoint-only sampling does not.
    with torch.no_grad():
        _ = handler._model.ts_to_img(reference_tensor[:1].to(args.device))

    dummy_ts = torch.zeros(
        sample_count,
        int(args.seq_len),
        class_metadata["channels"],
        dtype=torch.float32,
    )

    with torch.no_grad():
        generated_set = handler.sample(
            sample_count,
            class_label,
            class_metadata,
            (dummy_ts, embeddings),
        ).detach().cpu()

    real_ts = reference_tensor.float()
    real_set_ntc = _to_ntc(real_ts)
    generated_set_ntc = _to_ntc(generated_set)
    scores = evaluate_model_uncond(
        real_set_ntc,
        generated_set_ntc,
        dataset_name,
        args.device,
        eval_metrics=cli_args.eval_metrics,
        metric_iteration=cli_args.metric_iteration,
        base_path=cli_args.ts2vec_dir,
        vae_ckpt_root=cli_args.fid_vae_ckpt_root,
    )

    output_path = cli_args.output or default_output_path(args, dataset_name, cli_args.embeds_path)
    output_path = os.path.abspath(output_path)
    payload = {
        "dataset": dataset_name,
        "epoch": _extract_epoch_from_ckpt(cli_args.model_ckpt),
        "eval_split": cli_args.reference_split,
        "real_ts": real_ts,
        "sampled_ts": generated_set.float(),
        "model_ckpt": os.path.abspath(cli_args.model_ckpt),
        "embeds_path": os.path.abspath(cli_args.embeds_path),
    }
    _save_eval_samples(output_path, payload)

    summary = {
        "config": os.path.abspath(cli_args.config),
        "model_ckpt": os.path.abspath(cli_args.model_ckpt),
        "embeds_path": os.path.abspath(cli_args.embeds_path),
        "dataset": dataset_name,
        "reference_split": cli_args.reference_split,
        "num_samples": int(sample_count),
        "batch_size": int(args.batch_size),
        "seed": int(cli_args.seed),
        "device": str(args.device),
        "output": output_path,
        "sampled_ts_shape": list(generated_set.shape),
        "real_ts_shape": list(real_ts.shape),
    }
    summary.update(scores)

    if cli_args.save_metadata:
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    if cli_args.output_jsonl:
        output_jsonl = os.path.abspath(cli_args.output_jsonl)
        os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
        with open(output_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
