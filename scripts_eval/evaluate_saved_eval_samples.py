import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics import evaluate_model_uncond


EPOCH_PATTERN = re.compile(r"_epoch_(\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved eval_samples/*.pt files.")
    parser.add_argument("--eval-samples-dir", type=str, required=True, help="Directory that contains saved eval_samples .pt files.")
    parser.add_argument("--pattern", type=str, default="*.pt", help="Glob pattern for sample files.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-metrics", nargs="+", default=["disc", "contextFID", "pred", "vaeFID"])
    parser.add_argument("--metric-iteration", type=int, default=10)
    parser.add_argument("--ts2vec-dir", type=str, default=None, help="Cache dir for TS2Vec contextFID artifacts.")
    parser.add_argument("--fid-vae-ckpt-root", type=str, default=None, help="Root dir that stores fid_vae checkpoints.")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def _extract_epoch(path):
    match = EPOCH_PATTERN.search(path.name)
    return int(match.group(1)) if match else -1


def _to_ntc(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        x = tensor_or_array.detach().cpu().float()
    else:
        x = torch.as_tensor(tensor_or_array).detach().cpu().float()

    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor with shape (N, T, C) or (N, C, T), got {tuple(x.shape)}")

    # Normalize to (N, T, C) for evaluate_model_uncond.
    if x.shape[1] < x.shape[2]:
        x = x.permute(0, 2, 1)
    return x.numpy()


def _evaluate_single(real_set, fake_set, dataset, args):
    return evaluate_model_uncond(
        real_set,
        fake_set,
        dataset,
        args.device,
        eval_metrics=args.eval_metrics,
        metric_iteration=args.metric_iteration,
        base_path=args.ts2vec_dir,
        vae_ckpt_root=args.fid_vae_ckpt_root,
    )


def _evaluate_payload(payload, args):
    dataset = payload.get("dataset", "unknown")
    epoch = int(payload.get("epoch", -1))
    eval_split = payload.get("eval_split", "unknown")

    if "real_ts" not in payload or "sampled_ts" not in payload:
        raise KeyError("Expected payload to contain 'real_ts' and 'sampled_ts'.")

    real_set = _to_ntc(payload["real_ts"])
    sampled_ts = payload["sampled_ts"]

    result = {
        "dataset": dataset,
        "epoch": epoch,
        "eval_split": eval_split,
    }

    if torch.is_tensor(sampled_ts):
        sampled = sampled_ts.detach().cpu().float()
    else:
        sampled = torch.as_tensor(sampled_ts).detach().cpu().float()

    if sampled.ndim == 3:
        fake_set = _to_ntc(sampled)
        result.update(_evaluate_single(real_set, fake_set, dataset, args))
        result["num_variants"] = 1
        return result

    if sampled.ndim == 4:
        variant_scores = []
        for variant_idx in range(sampled.shape[0]):
            fake_set = _to_ntc(sampled[variant_idx])
            scores = _evaluate_single(real_set, fake_set, dataset, args)
            scores["variant_idx"] = int(variant_idx)
            variant_scores.append(scores)

        result["num_variants"] = int(sampled.shape[0])
        result["variant_scores"] = variant_scores

        metric_keys = [key for key in variant_scores[0].keys() if key != "variant_idx"]
        for key in metric_keys:
            values = [scores[key] for scores in variant_scores]
            result[key] = float(np.mean(values))
        return result

    raise ValueError(f"Expected sampled_ts to have ndim 3 or 4, got shape {tuple(sampled.shape)}")


def _default_output_paths(eval_samples_dir):
    output_dir = Path(eval_samples_dir)
    return (
        output_dir / "eval_metrics_summary.csv",
        output_dir / "eval_metrics_summary.json",
    )


def main():
    args = parse_args()
    eval_samples_dir = Path(args.eval_samples_dir)
    if not eval_samples_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {eval_samples_dir}")

    if args.ts2vec_dir is None and "contextFID" in args.eval_metrics:
        args.ts2vec_dir = str(eval_samples_dir / "TS2VEC")
    if args.ts2vec_dir is not None and "contextFID" in args.eval_metrics:
        os.makedirs(args.ts2vec_dir, exist_ok=True)

    default_csv, default_json = _default_output_paths(eval_samples_dir)
    output_csv = Path(args.output_csv) if args.output_csv else default_csv
    output_json = Path(args.output_json) if args.output_json else default_json

    sample_paths = sorted(eval_samples_dir.glob(args.pattern), key=lambda p: (_extract_epoch(p), p.name))
    if not sample_paths:
        raise FileNotFoundError(f"No files matched pattern '{args.pattern}' in {eval_samples_dir}")

    all_results = []
    for sample_path in sample_paths:
        payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        result = _evaluate_payload(payload, args)
        result["sample_path"] = str(sample_path)
        all_results.append(result)
        print(json.dumps(result, indent=2))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for result in all_results:
        row = {}
        for key, value in result.items():
            if key == "variant_scores":
                continue
            row[key] = value
        csv_rows.append(row)

    fieldnames = []
    for row in csv_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved CSV summary to {output_csv}")
    print(f"Saved JSON summary to {output_json}")


if __name__ == "__main__":
    main()
