import argparse
import json
import logging
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
    parser.add_argument("--prior-ckpt", type=str, default=None)
    parser.add_argument("--sample-source", type=str, default=None, choices=["prior", "posterior", "both"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--save-generated-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-metrics", nargs="+", type=str, default=None)
    parser.add_argument("--ts2vec-dir", type=str, default=None)
    parser.add_argument("--no-ema-eval", action="store_true")
    parser.add_argument("--posterior-noise-std", type=float, default=0.0)
    parser.add_argument("--posterior-noise-alpha", type=float, default=0.0)
    parser.add_argument("--prior-sampling-method", type=str, default=None)
    parser.add_argument("--prior-num-steps", type=int, default=None)
    parser.add_argument("--prior-atol", type=float, default=None)
    parser.add_argument("--prior-rtol", type=float, default=None)
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
    elif split == "test":
        dataset = get_test(dataset_config)
    else:
        raise ValueError(f"Unsupported split: {split}")
    return dataset_to_tensor(dataset, args)


def maybe_slice_tensor(tensor, max_samples):
    if max_samples is None:
        return tensor
    return tensor[:max_samples]


def make_default_output_json(dataset, split, sample_source, conditional_ckpt, prior_ckpt):
    results_dir = os.path.join(
        "./logs",
        "self_conditional_generator",
        "eval",
        dataset,
    )
    os.makedirs(results_dir, exist_ok=True)
    cond_tag = os.path.splitext(os.path.basename(conditional_ckpt))[0]
    prior_tag = os.path.splitext(os.path.basename(prior_ckpt))[0] if prior_ckpt else "no_prior"
    filename = f"{split}_{sample_source}_{cond_tag}__{prior_tag}.json"
    return os.path.join(results_dir, filename)


def generate_posterior_variants(
    handler,
    eval_tensor,
    seq_len,
    n_var,
    batch_size,
    noise_std,
    noise_alpha,
    base_seed,
):
    clean_batches = []
    noisy_batches = []
    device = handler.device
    use_noisy = (noise_std > 0.0) or (noise_alpha > 0.0)

    with handler._model.ema_scope(), torch.no_grad():
        for batch_start in range(0, len(eval_tensor), batch_size):
            batch_idx = batch_start // batch_size
            batch = eval_tensor[batch_start: batch_start + batch_size].to(device=device, dtype=torch.float32)
            context_trend, context_coarse_seasonal, context_seasonal = handler._model.multi_scale_vae.ts_to_z(
                batch, sample=False
            )
            context = torch.cat([context_trend, context_coarse_seasonal, context_seasonal], dim=-1)
            context = context.permute(0, 2, 1)

            torch.manual_seed(base_seed + batch_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(base_seed + batch_idx)
            clean_batches.append(
                handler._model.generate_from_context(context, seq_len, n_var)
            )

            if use_noisy:
                gaussian = torch.randn_like(context)
                if noise_alpha > 0.0:
                    noisy_context = (1.0 - noise_alpha) * context + noise_alpha * gaussian
                else:
                    noisy_context = context + gaussian * noise_std
                torch.manual_seed(base_seed + batch_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(base_seed + batch_idx)
                noisy_batches.append(
                    handler._model.generate_from_context(noisy_context, seq_len, n_var)
                )

    outputs = {"posterior": torch.cat(clean_batches, dim=0)}
    if use_noisy:
        if noise_alpha > 0.0:
            variant_name = f"posterior_noisy_alpha_{noise_alpha:g}"
        else:
            variant_name = f"posterior_noisy_std_{noise_std:g}"
        outputs[variant_name] = torch.cat(noisy_batches, dim=0)
    return outputs


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
    if cli_args.prior_ckpt is not None:
        args.prior_ckpt = cli_args.prior_ckpt
    if cli_args.sample_source is not None:
        args.sample_source = cli_args.sample_source
    if any(
        value is not None
        for value in (
            cli_args.prior_sampling_method,
            cli_args.prior_num_steps,
            cli_args.prior_atol,
            cli_args.prior_rtol,
        )
    ):
        sampler = dict(getattr(args, "sampler", {}) or {})
        params = dict(sampler.get("params", {}) or {})
        sampler["mode"] = sampler.get("mode", "ODE")
        if cli_args.prior_sampling_method is not None:
            params["sampling_method"] = cli_args.prior_sampling_method
        if cli_args.prior_num_steps is not None:
            params["num_steps"] = int(cli_args.prior_num_steps)
        if cli_args.prior_atol is not None:
            params["atol"] = float(cli_args.prior_atol)
        if cli_args.prior_rtol is not None:
            params["rtol"] = float(cli_args.prior_rtol)
        sampler["params"] = params
        args.sampler = sampler

    if not getattr(args, "resume_ckpt", None):
        raise ValueError("A finetuned conditional generator checkpoint must be provided via config or --conditional-ckpt.")
    if args.sample_source in {"prior", "both"} and not getattr(args, "prior_ckpt", None):
        raise ValueError("A finetuned diffusion prior checkpoint must be provided for prior sampling.")

    torch.random.manual_seed(args.seed)
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

    if (cli_args.posterior_noise_std > 0.0 or cli_args.posterior_noise_alpha > 0.0) and args.sample_source in {"posterior", "both"}:
        posterior_variants = generate_posterior_variants(
            handler=handler,
            eval_tensor=eval_tensor,
            seq_len=args.seq_len,
            n_var=class_metadata["channels"],
            batch_size=int(getattr(args, "batch_size", 128)),
            noise_std=float(cli_args.posterior_noise_std),
            noise_alpha=float(cli_args.posterior_noise_alpha),
            base_seed=int(args.seed),
        )
        generated_sets["posterior"] = posterior_variants["posterior"]
        for key, value in posterior_variants.items():
            if key != "posterior":
                generated_sets[key] = value

    real_set = eval_tensor.cpu().detach().numpy()
    ts2vec_dir = cli_args.ts2vec_dir or os.path.join("./logs", "TS2VEC")
    os.makedirs(ts2vec_dir, exist_ok=True)

    all_scores = {}
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
            save_path = os.path.join(cli_args.save_generated_dir, f"{dataset_name}_{cli_args.split}_{variant_name}.npy")
            np.save(save_path, generated_np)

    result = {
        "dataset": dataset_name,
        "split": cli_args.split,
        "num_samples": int(len(eval_tensor)),
        "sample_source": args.sample_source,
        "use_ema_for_eval": not cli_args.no_ema_eval,
        "posterior_noise_std": float(cli_args.posterior_noise_std),
        "posterior_noise_alpha": float(cli_args.posterior_noise_alpha),
        "prior_sampler": getattr(args, "sampler", None),
        "conditional_ckpt": args.resume_ckpt,
        "prior_ckpt": getattr(args, "prior_ckpt", None),
        "metrics": all_scores,
    }

    output_json = cli_args.output_json or make_default_output_json(
        dataset_name,
        cli_args.split,
        args.sample_source,
        args.resume_ckpt,
        getattr(args, "prior_ckpt", None),
    )
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved metrics to {output_json}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
