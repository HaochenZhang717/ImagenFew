import argparse
import json
import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from stage1_dataset import load_split_arrays
from train_stage2 import (
    build_backbone,
    build_stage1_decoder,
    build_transport_and_sampler,
    sample_and_decode,
    to_plain_dict,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sample captions from a trained Stage 2 diffusion prior.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to stage2 checkpoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional stage2 config path. If omitted, use the config stored in checkpoint.",
    )
    parser.add_argument(
        "--stage1-config",
        type=str,
        default=None,
        help="Optional path to stage1 config. Overrides stage1.config_path from the stage2 config.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default=None,
        help="Optional path to stage1 checkpoint. Overrides stage1.checkpoint_path from the stage2 config.",
    )
    parser.add_argument("--override", nargs="*", default=[], help="OmegaConf dotlist overrides")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of captions to sample. If omitted, sample the same number as the test split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights from stage2 checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--output-npy",
        type=str,
        default=None,
        help="Optional path to save generated captions in VerbalTSDatasets .npy format.",
    )
    parser.add_argument(
        "--retrieve-train",
        action="store_true",
        help="Enable retrieval of nearest training captions for each generated caption.",
    )
    parser.add_argument(
        "--retrieve-train-topk",
        type=int,
        default=5,
        help="Top-k nearest training captions to retrieve when --retrieve-train is enabled.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Embedding model used for train-caption retrieval.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=4)
    parser.add_argument("--embedding-max-length", type=int, default=8192)
    parser.add_argument(
        "--train-embedding-cache",
        type=str,
        default=None,
        help="Optional path to cache training caption embeddings.",
    )
    return parser.parse_args()


def to_verbalts_caption_array(captions) -> np.ndarray:
    return np.asarray(captions, dtype=str).reshape(-1, 1)


def resolve_default_npy_output_path(json_output_path: str) -> str:
    root, _ = os.path.splitext(os.path.abspath(json_output_path))
    return f"{root}_text_caps.npy"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_config(args, checkpoint: Dict) -> Dict:
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        if "config" in checkpoint:
            checkpoint_cfg = to_plain_dict(OmegaConf.create(checkpoint["config"]))
            for key in ("seq_len", "token_dim"):
                if "diffusion_model" in checkpoint_cfg and key in checkpoint_cfg["diffusion_model"]:
                    cfg.diffusion_model[key] = checkpoint_cfg["diffusion_model"][key]
    elif "config" in checkpoint:
        cfg = OmegaConf.create(checkpoint["config"])
    else:
        raise ValueError("No config provided and checkpoint does not contain an embedded config.")

    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))
    cfg = to_plain_dict(cfg)

    if args.num_samples is not None:
        cfg["sampling"]["num_decode_samples"] = args.num_samples

    return cfg


def align_diffusion_shape_from_checkpoint(cfg: Dict, checkpoint: Dict) -> None:
    state_dict = checkpoint.get("ema") or checkpoint["model"]
    if "pos_embed" in state_dict:
        cfg["diffusion_model"]["seq_len"] = int(state_dict["pos_embed"].shape[1])

    if "x_embedder.proj.weight" in state_dict:
        cfg["diffusion_model"]["token_dim"] = int(state_dict["x_embedder.proj.weight"].shape[1])


def resolve_device(cfg: Dict) -> torch.device:
    requested_device = cfg.get("device", "auto")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA but it is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


@torch.no_grad()
def encode_texts(
    texts,
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    desc: str,
) -> torch.Tensor:
    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch_texts = texts[start : start + batch_size]
        batch_dict = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def resolve_train_embedding_cache_path(args, dataset_root: str) -> str | None:
    if args.train_embedding_cache:
        return os.path.abspath(args.train_embedding_cache)
    if not args.retrieve_train or args.retrieve_train_topk <= 0:
        return None
    safe_model_name = args.embedding_model.replace("/", "__")
    return os.path.abspath(
        os.path.join(dataset_root, f"train_caption_embeddings_{safe_model_name}.pt")
    )


@torch.no_grad()
def retrieve_training_captions(args, dataset_root: str, generated_captions, device: torch.device):
    _, train_captions, _ = load_split_arrays(dataset_root, "train")
    cache_path = resolve_train_embedding_cache_path(args, dataset_root)

    embed_tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, padding_side="left")
    embed_model = AutoModel.from_pretrained(args.embedding_model).to(device)
    embed_model.eval()

    task = "Given a generated time-series caption, retrieve semantically similar training captions"
    query_texts = [get_detailed_instruct(task, text) for text in generated_captions]

    if cache_path is not None and os.path.exists(cache_path):
        cache = torch.load(cache_path, map_location="cpu")
        train_embeddings = cache["embeddings"].float()
        cached_train_captions = cache["captions"]
        if cached_train_captions != train_captions:
            raise ValueError(
                f"Training caption cache mismatch for {cache_path}. Delete it or provide a new cache path."
            )
    else:
        train_embeddings = encode_texts(
            texts=train_captions,
            tokenizer=embed_tokenizer,
            model=embed_model,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            desc="embed train captions",
        )
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            torch.save(
                {
                    "captions": train_captions,
                    "embeddings": train_embeddings,
                    "embedding_model": args.embedding_model,
                },
                cache_path,
            )

    query_embeddings = encode_texts(
        texts=query_texts,
        tokenizer=embed_tokenizer,
        model=embed_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        desc="embed generated captions",
    )

    scores = query_embeddings @ train_embeddings.T
    topk_scores, topk_indices = torch.topk(scores, k=min(args.retrieve_train_topk, scores.shape[1]), dim=1)

    retrieval_rows = []
    for query_idx, query_text in enumerate(generated_captions):
        retrieved = []
        for rank, (score, train_idx) in enumerate(zip(topk_scores[query_idx], topk_indices[query_idx]), start=1):
            retrieved.append(
                {
                    "rank": rank,
                    "train_index": int(train_idx.item()),
                    "score": float(score.item()),
                    "text": train_captions[int(train_idx.item())],
                }
            )
        retrieval_rows.append(
            {
                "sample_id": query_idx,
                "generated_caption": query_text,
                "retrieved_train_captions": retrieved,
            }
        )

    return {
        "embedding_model": args.embedding_model,
        "topk": args.retrieve_train_topk,
        "train_caption_count": len(train_captions),
        "cache_path": cache_path,
        "results": retrieval_rows,
    }


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = resolve_config(args, checkpoint)
    align_diffusion_shape_from_checkpoint(cfg, checkpoint)
    device = resolve_device(cfg)
    set_seed(args.seed)

    if args.stage1_config is not None:
        cfg["stage1"]["config_path"] = args.stage1_config
    if args.stage1_checkpoint is not None:
        cfg["stage1"]["checkpoint_path"] = args.stage1_checkpoint

    stage1_cfg = to_plain_dict(OmegaConf.load(cfg["stage1"]["config_path"]))
    stage1_ckpt = torch.load(cfg["stage1"]["checkpoint_path"], map_location="cpu")
    cfg["stage1_prompt"] = stage1_cfg["data"]["prompt_template"]
    dataset_root = cfg["data"].get("dataset_root", stage1_cfg["data"]["dataset_root"])

    if args.num_samples is not None:
        cfg["sampling"]["num_decode_samples"] = args.num_samples
    else:
        test_ts, _, _ = load_split_arrays(dataset_root, "test")
        cfg["sampling"]["num_decode_samples"] = int(len(test_ts))

    model = build_backbone(cfg).to(device)
    state_key = "ema" if args.use_ema and "ema" in checkpoint else "model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.eval()

    _, sample_fn = build_transport_and_sampler(cfg)
    decoder, tokenizer = build_stage1_decoder(stage1_cfg, stage1_ckpt, device=device)

    sampled_latents, decoded = sample_and_decode(
        model_for_sampling=model,
        decoder=decoder,
        tokenizer=tokenizer,
        sample_fn=sample_fn,
        cfg=cfg,
        device=device,
    )

    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "state_dict_used": state_key,
        "stage1_config_path": os.path.abspath(cfg["stage1"]["config_path"]),
        "stage1_checkpoint_path": os.path.abspath(cfg["stage1"]["checkpoint_path"]),
        "num_samples": len(decoded),
        "dataset_root": dataset_root,
        "matched_split": "test" if args.num_samples is None else None,
        "prompt": cfg["stage1_prompt"],
        "captions": [{"sample_id": idx, "text": text} for idx, text in enumerate(decoded)],
        "latent_shape": list(sampled_latents.shape),
    }

    if args.retrieve_train:
        result["train_caption_retrieval"] = retrieve_training_captions(
            args=args,
            dataset_root=dataset_root,
            generated_captions=decoded,
            device=device,
        )

    verbalts_caption_array = to_verbalts_caption_array(decoded)

    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)

        if args.output_npy is None:
            np.save(resolve_default_npy_output_path(output_path), verbalts_caption_array)

    if args.output_npy:
        output_npy_path = os.path.abspath(args.output_npy)
        os.makedirs(os.path.dirname(output_npy_path) or ".", exist_ok=True)
        np.save(output_npy_path, verbalts_caption_array)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
