import argparse
import json
import os

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from caption_generator.stage1_dataset import compute_train_normalization_stats, load_split_arrays
from caption_generator.stage1_model import Stage1LatentCaptionModel


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions with a trained Stage 1 model.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)

    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

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

    train_ts, _, _ = load_split_arrays(cfg["data"]["dataset_root"], "train")
    stats = compute_train_normalization_stats(train_ts)
    ts, captions, _ = load_split_arrays(cfg["data"]["dataset_root"], args.split)

    sample = (ts[args.index] - stats["mean"]) / stats["std"]
    sample = torch.from_numpy(sample.T.copy()).unsqueeze(0).float().to(device)
    prompt = cfg["data"]["prompt_template"]
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    outputs = model.generate(
        ts=sample,
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
    pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(
        json.dumps(
            {
                "split": args.split,
                "index": args.index,
                "prompt": prompt,
                "prediction": pred,
                "ground_truth": captions[args.index],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
