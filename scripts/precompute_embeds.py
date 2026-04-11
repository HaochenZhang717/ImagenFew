import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from qwen3_vl_embedding import Qwen3VLEmbedder


DEFAULT_HF_CACHE = "/playpen-shared/haochenz/hf_cache"
IMAGE_NAME_PATTERN = re.compile(r"image(\d+)_ch(\d+)\.png$")


os.environ.setdefault("HF_HOME", DEFAULT_HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", DEFAULT_HF_CACHE)
os.environ.setdefault("HF_DATASETS_CACHE", DEFAULT_HF_CACHE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge caption parts, sort by image/channel, and save caption embeddings as an (N, C, D) tensor."
    )
    parser.add_argument(
        "--caption-dir",
        type=str,
        required=True,
        help="Directory containing caption jsonl parts like train_caps_0_4.jsonl ... train_caps_3_4.jsonl",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split prefix used in caption files, e.g. train/test/valid.",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=4,
        help="Expected number of caption part files.",
    )
    parser.add_argument(
        "--n-vars",
        type=int,
        required=True,
        help="Number of variables/channels per sample.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Output .pt path. The saved object is a tensor with shape (N, n_vars, dim).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for text embedding.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Embedding model name or path.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype passed to the embedding model.",
    )
    return parser.parse_args()


def resolve_torch_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_caption_rows(caption_dir: str, split: str, num_parts: int) -> Dict[str, str]:
    image_to_caption: Dict[str, str] = {}

    for part_id in range(num_parts):
        path = os.path.join(caption_dir, f"{split}_caps_{part_id}_{num_parts}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing caption part file: {path}")

        rows = load_jsonl(path)
        for row in rows:
            image_name = row["image"]
            caption = row["caption"]
            if image_name in image_to_caption and image_to_caption[image_name] != caption:
                raise ValueError(f"Found conflicting captions for {image_name}")
            image_to_caption[image_name] = caption

    if not image_to_caption:
        raise ValueError(f"No captions were loaded from {caption_dir}")

    return image_to_caption


def parse_image_name(image_name: str) -> Tuple[int, int]:
    match = IMAGE_NAME_PATTERN.fullmatch(image_name)
    if match is None:
        raise ValueError(f"Unexpected image name format: {image_name}")
    image_id = int(match.group(1))
    ch_id = int(match.group(2))
    return image_id, ch_id


def build_sorted_caption_matrix(image_to_caption: Dict[str, str], n_vars: int) -> Tuple[List[List[str]], List[int]]:
    by_sample: Dict[int, Dict[int, str]] = {}

    for image_name, caption in image_to_caption.items():
        image_id, ch_id = parse_image_name(image_name)
        by_sample.setdefault(image_id, {})
        if ch_id in by_sample[image_id]:
            raise ValueError(f"Duplicate caption for sample {image_id}, channel {ch_id}")
        by_sample[image_id][ch_id] = caption

    ordered_ids = sorted(by_sample.keys())
    caption_matrix: List[List[str]] = []
    incomplete_ids = []

    for image_id in ordered_ids:
        sample_caps = by_sample[image_id]
        if len(sample_caps) != n_vars:
            incomplete_ids.append(image_id)
            continue
        caption_matrix.append([sample_caps[ch] for ch in range(n_vars)])

    if incomplete_ids:
        preview = incomplete_ids[:10]
        raise ValueError(
            f"Found {len(incomplete_ids)} incomplete samples. "
            f"Example sample ids: {preview}"
        )

    return caption_matrix, ordered_ids


def embed_texts(texts: List[str], batch_size: int, model_name: str, torch_dtype) -> torch.Tensor:
    model = Qwen3VLEmbedder(
        model_name_or_path=model_name,
        torch_dtype=torch_dtype,
    )

    all_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding captions"):
        batch_texts = texts[start:start + batch_size]
        inputs = [{"text": text} for text in batch_texts]
        embeds = model.process(inputs)
        all_embeddings.append(embeds.cpu())

    return torch.cat(all_embeddings, dim=0)


def main():
    args = parse_args()

    print("Loading caption parts...")
    image_to_caption = collect_caption_rows(
        caption_dir=args.caption_dir,
        split=args.split,
        num_parts=args.num_parts,
    )
    print(f"Loaded {len(image_to_caption)} caption rows.")

    print("Sorting and grouping captions...")
    caption_matrix, ordered_ids = build_sorted_caption_matrix(
        image_to_caption=image_to_caption,
        n_vars=args.n_vars,
    )
    num_samples = len(caption_matrix)
    print(f"Grouped into {num_samples} complete samples.")

    flat_texts = [text for sample_caps in caption_matrix for text in sample_caps]
    print(f"Embedding {len(flat_texts)} captions...")
    embeddings = embed_texts(
        texts=flat_texts,
        batch_size=args.batch_size,
        model_name=args.model_name,
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )

    emb_dim = embeddings.shape[-1]
    embeddings = embeddings.reshape(num_samples, args.n_vars, emb_dim).contiguous()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(embeddings, args.save_path)

    print(f"Saved tensor to: {args.save_path}")
    print(f"Tensor shape: {tuple(embeddings.shape)}")
    print(f"First sample id: {ordered_ids[0]}")
    print(f"Last sample id: {ordered_ids[-1]}")


if __name__ == "__main__":
    main()
