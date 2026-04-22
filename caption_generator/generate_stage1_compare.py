import argparse
import json
import os
import re
import sys
from contextlib import nullcontext
from difflib import SequenceMatcher
from typing import Dict, List

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from caption_generator.stage1_dataset import compute_train_normalization_stats, load_split_arrays
from caption_generator.stage1_model import Stage1LatentCaptionModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate captions from time series for a dataset split and compare them with ground-truth captions."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--output-summary", type=str, default=None)
    return parser.parse_args()


def resolve_device(cfg: Dict) -> torch.device:
    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def maybe_strip_prompt(text: str, prompt: str) -> str:
    normalized_text = normalize_text(text)
    normalized_prompt = normalize_text(prompt)
    if normalized_text.startswith(normalized_prompt):
        stripped = normalized_text[len(normalized_prompt) :].lstrip(": \n")
        if stripped:
            return stripped
    return normalized_text


def compute_text_metrics(prediction: str, ground_truth: str) -> Dict[str, float | bool]:
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    pred_lower = pred_norm.lower()
    gt_lower = gt_norm.lower()
    return {
        "exact_match": pred_norm == gt_norm,
        "case_insensitive_exact_match": pred_lower == gt_lower,
        "char_similarity": SequenceMatcher(None, pred_norm, gt_norm).ratio(),
        "pred_num_chars": len(pred_norm),
        "gt_num_chars": len(gt_norm),
        "pred_num_lines": pred_norm.count("\n") + (1 if pred_norm else 0),
        "gt_num_lines": gt_norm.count("\n") + (1 if gt_norm else 0),
    }


def main():
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    device = resolve_device(cfg)
    model_dtype_name = cfg.get("model", {}).get("torch_dtype")
    use_amp = device.type == "cuda" and model_dtype_name in {"float16", "bfloat16"}
    amp_dtype = getattr(torch, model_dtype_name) if use_amp else None

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["llm_name"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        use_fast=cfg["model"].get("use_fast_tokenizer", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Stage1LatentCaptionModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    dataset_root = cfg["data"]["dataset_root"]
    prompt = cfg["data"]["prompt_template"]

    train_ts, _, _ = load_split_arrays(dataset_root, "train")
    stats = compute_train_normalization_stats(train_ts)
    ts, captions, attrs = load_split_arrays(dataset_root, args.split)

    if args.max_samples is not None:
        ts = ts[: args.max_samples]
        captions = captions[: args.max_samples]
        if attrs is not None:
            attrs = attrs[: args.max_samples]

    normalized_ts = (ts - stats["mean"]) / stats["std"]
    batch_size = args.batch_size or cfg.get("training", {}).get("eval_batch_size", 8)

    output_jsonl = os.path.abspath(args.output_jsonl)
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    total = len(captions)
    num_exact = 0
    num_case_exact = 0
    similarity_sum = 0.0

    with open(output_jsonl, "w", encoding="utf-8") as fp:
        for start in tqdm(range(0, total, batch_size), desc=f"generate-{args.split}"):
            end = min(start + batch_size, total)
            batch_ts = torch.from_numpy(normalized_ts[start:end].transpose(0, 2, 1).copy()).float().to(device)
            prompts = [prompt] * (end - start)
            tokenized = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
            )
            with torch.no_grad(), autocast_ctx:
                outputs = model.generate(
                    ts=batch_ts,
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    generation_kwargs={
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": args.temperature > 0,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    },
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            cleaned_predictions: List[str] = [maybe_strip_prompt(text, prompt) for text in decoded]

            for local_idx, prediction in enumerate(cleaned_predictions):
                sample_id = start + local_idx
                ground_truth = captions[sample_id]
                metrics = compute_text_metrics(prediction, ground_truth)
                num_exact += int(metrics["exact_match"])
                num_case_exact += int(metrics["case_insensitive_exact_match"])
                similarity_sum += float(metrics["char_similarity"])

                record = {
                    "split": args.split,
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "metrics": metrics,
                }
                if attrs is not None:
                    record["attrs_idx"] = attrs[sample_id].tolist()
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset_root": os.path.abspath(dataset_root),
        "split": args.split,
        "output_jsonl": output_jsonl,
        "num_samples": total,
        "batch_size": batch_size,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "metrics": {
            "exact_match_rate": num_exact / total if total else 0.0,
            "case_insensitive_exact_match_rate": num_case_exact / total if total else 0.0,
            "avg_char_similarity": similarity_sum / total if total else 0.0,
        },
    }

    if args.output_summary is not None:
        summary_path = os.path.abspath(args.output_summary)
    else:
        root, _ = os.path.splitext(output_jsonl)
        summary_path = f"{root}.summary.json"
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
