import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _flatten_caption_array(captions: np.ndarray) -> List[str]:
    flattened: List[str] = []
    for item in captions:
        if isinstance(item, np.ndarray):
            flattened.append(str(item.reshape(-1)[0]))
        elif isinstance(item, (list, tuple)):
            flattened.append(str(item[0]))
        else:
            flattened.append(str(item))
    return flattened


def compute_train_normalization_stats(train_ts: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(train_ts.mean()),
        "std": float(train_ts.std() + 1e-6),
    }


def load_split_arrays(dataset_root: str, split: str):
    ts = np.load(os.path.join(dataset_root, f"{split}_ts.npy"), allow_pickle=True).astype(np.float32)
    captions = np.load(
        os.path.join(dataset_root, f"{split}_text_caps.npy"),
        allow_pickle=True,
    )
    attrs_path = os.path.join(dataset_root, f"{split}_attrs_idx.npy")
    attrs = np.load(attrs_path, allow_pickle=True) if os.path.exists(attrs_path) else None
    return ts, _flatten_caption_array(captions), attrs


class TimeSeriesCaptionDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str,
        prompt_template: str,
        normalization_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.prompt_template = prompt_template

        ts, captions, attrs = load_split_arrays(dataset_root, split)
        if normalization_stats is None:
            normalization_stats = compute_train_normalization_stats(ts)
        self.normalization_stats = normalization_stats

        self.ts = (ts - normalization_stats["mean"]) / normalization_stats["std"]
        self.captions = captions
        self.attrs = attrs

    def __len__(self) -> int:
        return len(self.ts)

    def __getitem__(self, index: int) -> Dict[str, object]:
        ts = self.ts[index]
        if ts.ndim != 2:
            raise ValueError(f"Expected a 2D time-series sample, got shape {ts.shape}")

        sample = {
            "ts": torch.from_numpy(ts.T.copy()).float(),
            "prompt_text": self.prompt_template,
            "caption_text": self.captions[index].strip(),
        }
        if self.attrs is not None:
            sample["attrs_idx"] = torch.as_tensor(self.attrs[index], dtype=torch.long)
        return sample


@dataclass
class CaptionCollator:
    tokenizer: object
    max_prompt_length: int
    max_caption_length: int
    caption_transform: Optional[Callable[[str], str]] = None

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt_text"] for item in batch]
        raw_captions = [item["caption_text"] for item in batch]
        if self.caption_transform is None:
            captions = raw_captions
        else:
            captions = [self.caption_transform(caption) for caption in raw_captions]
        ts = torch.stack([item["ts"] for item in batch], dim=0)

        prompt_tokens = self.tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        caption_tokens = self.tokenizer(
            captions,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_caption_length,
        )

        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer must define a pad_token_id before batching.")

        input_ids: List[List[int]] = []
        labels: List[List[int]] = []
        attention_mask: List[List[int]] = []

        for prompt_ids, caption_ids in zip(prompt_tokens["input_ids"], caption_tokens["input_ids"]):
            seq: List[int] = []
            seq_labels: List[int] = []

            if bos_id is not None:
                seq.append(bos_id)
                seq_labels.append(-100)

            seq.extend(prompt_ids)
            seq_labels.extend([-100] * len(prompt_ids))

            seq.extend(caption_ids)
            seq_labels.extend(caption_ids)

            if eos_id is not None:
                seq.append(eos_id)
                seq_labels.append(eos_id)

            input_ids.append(seq)
            labels.append(seq_labels)
            attention_mask.append([1] * len(seq))

        max_len = max(len(ids) for ids in input_ids)
        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, seq_labels, seq_mask in zip(input_ids, labels, attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(seq_labels + [-100] * pad_len)
            padded_attention_mask.append(seq_mask + [0] * pad_len)

        output = {
            "ts": ts,
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "raw_prompts": prompts,
            "raw_captions": raw_captions,
            "tokenized_captions": captions,
        }

        if "attrs_idx" in batch[0]:
            output["attrs_idx"] = torch.stack([item["attrs_idx"] for item in batch], dim=0)

        return output


def save_normalization_stats(stats: Dict[str, float], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
